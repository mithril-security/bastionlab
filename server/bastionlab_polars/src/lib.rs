use bastionlab_common::prelude::*;
use bastionlab_common::{
    array_store::ArrayStore,
    session::SessionManager,
    session_proto::ClientInfo,
    telemetry::{self, TelemetryEventProps},
    tracking::Tracking,
};

use polars::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{create_dir, read_dir, OpenOptions};
use std::io::{Error, ErrorKind};
use std::{future::Future, pin::Pin, time::Instant};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use utils::sanitize_df;
use uuid::Uuid;

pub mod polars_proto {
    tonic::include_proto!("bastionlab_polars");
}

use polars_proto::{
    polars_service_server::PolarsService, Empty, FetchChunk, Query, ReferenceList,
    ReferenceRequest, ReferenceResponse, SendChunk, SplitRequest,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

mod visitable;

pub mod access_control;
use access_control::*;

pub mod utils;

pub mod prelude {
    pub use bastionlab_common::prelude::*;
}

pub enum FetchStatus {
    Ok,
    Pending(String),
    Warning(String),
}

/// This a DataFrame intended to be streamed to the client.
/// It can be delayed when the data owner's approval is required.
pub struct DelayedDataFrame {
    future: Pin<Box<dyn Future<Output = Result<DataFrame, Status>> + Send>>,
    fetch_status: FetchStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFrameArtifact {
    dataframe: DataFrame,
    policy: Policy,
    fetchable: VerificationResult,
    blacklist: Vec<String>,
    query_details: String,
}

impl DataFrameArtifact {
    pub fn new(df: DataFrame, policy: Policy, blacklist: Vec<String>) -> Self {
        DataFrameArtifact {
            dataframe: df,
            policy,
            fetchable: VerificationResult::Unsafe {
                action: UnsafeAction::Reject,
                reason: String::from("DataFrames uploaded by the Data Owner are protected."),
            },
            blacklist,
            query_details: String::from("uploaded dataframe"),
        }
    }

    pub fn with_fetchable(mut self, fetchable: VerificationResult) -> Self {
        self.fetchable = fetchable;
        self
    }

    pub fn inherit(&self, df: DataFrame) -> Self {
        Self {
            dataframe: df,
            policy: self.policy.clone(),
            blacklist: self.blacklist.clone(),
            fetchable: self.fetchable.clone(),
            query_details: self.query_details.clone(),
        }
    }
}

#[derive(Clone)]
pub struct BastionLabPolars {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
    arrays: Arc<RwLock<HashMap<String, ArrayStore>>>,
    sess_manager: Arc<SessionManager>,
    tracking: Arc<Tracking>,
}

impl BastionLabPolars {
    pub fn new(sess_manager: Arc<SessionManager>, tracking: Arc<Tracking>) -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            arrays: Arc::new(RwLock::new(HashMap::new())),
            sess_manager,
            tracking,
        }
    }

    fn get_df(
        &self,
        identifier: &str,
        client_info: Option<ClientInfo>,
    ) -> Result<DelayedDataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        let artifact = dfs.get(identifier).ok_or_else(|| {
            Status::not_found(format!(
                "Could not find dataframe: identifier={}",
                identifier
            ))
        })?;
        if let VerificationResult::Unsafe { reason, .. } = &artifact.fetchable {
            println!(
                "Safe zone violation: a DataFrame has been non-privately fetched.
Reason: {}",
                reason
            );
        }
        Ok(match &artifact.fetchable {
            VerificationResult::Safe
            | VerificationResult::Unsafe {
                action: UnsafeAction::Log,
                ..
            } => {
                let mut df = artifact.dataframe.clone();
                sanitize_df(&mut df, &artifact.blacklist);
                telemetry::add_event(
                    TelemetryEventProps::FetchDataFrame {
                        dataset_name: Some(identifier.to_owned()),
                        request_accepted: true,
                    },
                    client_info,
                );
                DelayedDataFrame {
                    future: Box::pin(async { Ok(df) }),
                    fetch_status: if let VerificationResult::Unsafe {
                        action: UnsafeAction::Log,
                        reason,
                    } = &artifact.fetchable
                    {
                        FetchStatus::Warning(String::from(reason))
                    } else {
                        FetchStatus::Ok
                    },
                }
            }
            VerificationResult::Unsafe {
                action: UnsafeAction::Reject,
                reason,
            } => {
                let reason = reason.clone();
                DelayedDataFrame {
                    future: Box::pin(async move {
                        Err(Status::permission_denied(format!(
                        "Cannot fetch this DataFrame: operation denied by the data owner's policy
Reason: {}",
                        reason,
                    )))
                    }),
                    fetch_status: FetchStatus::Ok,
                }
            }
            VerificationResult::Unsafe {
                action: UnsafeAction::Review,
                reason,
            } => {
                let reason = reason.clone();
                let identifier = String::from(identifier);
                let query_details = artifact.query_details.clone();
                let dfs = Arc::clone(&self.dataframes);
                DelayedDataFrame {
                    fetch_status: FetchStatus::Pending(reason.clone()),
                    future: Box::pin(async move {
                        println!(
                            "A user requests unsafe access to one of your DataFrames
DataFrame identifier: {}
Reason the request is unsafe:
{}",
                            identifier, reason,
                        );

                        loop {
                            let mut ans = String::new();
                            println!("Accept [y], Reject [n], Show query details [s]?");
                            std::io::stdin()
                                .read_line(&mut ans)
                                .expect("Failed to read line");

                            match ans.trim() {
                                "y" => break,
                                "s" => {
                                    println!(
                                        "Query's Logical Plan:
{}",
                                        query_details,
                                    );
                                    continue;
                                }
                                "n" => {
                                    telemetry::add_event(
                                        TelemetryEventProps::FetchDataFrame {
                                            dataset_name: Some(identifier.to_owned()),
                                            request_accepted: false,
                                        },
                                        client_info,
                                    );
                                    return Err(Status::permission_denied(format!(
                                        "The data owner rejected the fetch operation.
Fetching a dataframe obtained with a non privacy-preserving query requires the approval of the data owner.
This dataframe was obtained in a non privacy-preserving fashion.
Reason: {}",
                                        reason
                                    )));
                                }
                                _ => continue,
                            }
                        }
                        telemetry::add_event(
                            TelemetryEventProps::FetchDataFrame {
                                dataset_name: Some(identifier.to_owned()),
                                request_accepted: true,
                            },
                            client_info,
                        );
                        Ok({
                            let guard = dfs.read().unwrap();
                            let artifact = guard.get(&identifier).ok_or_else(|| {
                                Status::not_found(format!(
                                    "Could not find dataframe: identifier={}",
                                    identifier
                                ))
                            })?;
                            let mut df = artifact.dataframe.clone();
                            sanitize_df(&mut df, &artifact.blacklist);
                            df
                        })
                    }),
                }
            }
        })
    }

    pub fn get_df_unchecked(&self, identifier: &str) -> Result<DataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        Ok(dfs
            .get(identifier)
            .ok_or_else(|| {
                Status::not_found(format!(
                    "Could not find dataframe: identifier={}",
                    identifier
                ))
            })?
            .dataframe
            .clone())
    }

    fn with_df_artifact_ref<T>(
        &self,
        identifier: &str,
        mut f: impl FnMut(&DataFrameArtifact) -> T,
    ) -> Result<T, Status> {
        let dfs = self.dataframes.read().unwrap();
        Ok(f(dfs.get(identifier).ok_or_else(|| {
            Status::not_found(format!(
                "Could not find dataframe: identifier={}",
                identifier
            ))
        })?))
    }

    pub fn get_header(&self, identifier: &str) -> Result<String, Status> {
        Ok(get_df_header(
            &self
                .dataframes
                .read()
                .unwrap()
                .get(identifier)
                .ok_or_else(|| {
                    Status::not_found(format!(
                        "Could not find dataframe: identifier={}",
                        identifier
                    ))
                })?
                .dataframe,
        )?)
    }

    fn get_headers(&self) -> Result<Vec<(String, String)>, Status> {
        let dataframes = self.dataframes.read().unwrap();
        let mut res = Vec::with_capacity(dataframes.len());
        for (k, v) in dataframes.iter() {
            let header = get_df_header(&v.dataframe)?;
            res.push((k.clone(), header));
        }
        Ok(res)
    }

    pub fn insert_df(&self, df: DataFrameArtifact, user_id: String) -> Result<String, Status> {
        let identifier = format!("{}", Uuid::new_v4());
        let size = df.dataframe.estimated_size();
        self.tracking
            .memory_quota_check(size, user_id, identifier.clone())?;
        let mut dfs = self.dataframes.write().unwrap();
        dfs.insert(identifier.clone(), df);
        Ok(identifier)
    }

    pub fn insert_array(&self, array: ArrayStore) -> String {
        let mut arrays = self.arrays.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        arrays.insert(identifier.clone(), array);
        identifier
    }

    pub fn get_array(&self, identifier: &str) -> Result<ArrayStore, Status> {
        let arrays = self.arrays.read().unwrap();
        let arr = arrays
            .get(identifier)
            .cloned()
            .ok_or(Status::invalid_argument(format!(
                "Could not find Array: {identifier}"
            )))?;
        Ok(arr)
    }

    fn persist_df(&self, identifier: &str) -> Result<(), Status> {
        let dataframes = self
            .dataframes
            .read()
            .map_err(|_| Status::internal("Unable to read dataframes!"))?;

        let df_artifact = dataframes
            .get(identifier)
            .ok_or_else(|| Status::not_found("Unable to find dataframe!"))?;

        if df_artifact.policy.check_savable() != true {
            return Err(Status::unknown("Dataframe is not savable"));
        }

        let error = create_dir("data_frames");
        match error {
            Ok(_) => {}
            Err(err) => {
                if err.kind() != ErrorKind::AlreadyExists {
                    return Err(Status::unknown(err.kind().to_string()));
                }
            }
        }

        let path = format!("data_frames/{}.json", identifier);
        let df_store = OpenOptions::new()
            .write(true)
            .create(true)
            .open(path)
            .map_err(|_| Status::internal("Unable to find or create storage file!"))?;

        serde_json::to_writer(df_store, df_artifact)
            .map_err(|_| Status::internal("Could not serialize dataframe artifact!"))?;

        Ok(())
    }

    pub fn load_dfs(&self) -> Result<(), Error> {
        let files = read_dir("data_frames")?;

        for file in files {
            let file = file?;
            let identifier = file.file_name().to_str().unwrap().replace(".json", "");

            let file = std::fs::OpenOptions::new()
                .read(true)
                .open(file.path().to_str().unwrap())?;
            let reader = std::io::BufReader::new(file);
            let df: DataFrameArtifact = serde_json::from_reader(reader)?;

            let mut dfs = self.dataframes.write().unwrap();
            dfs.insert(identifier, df);
        }
        Ok(())
    }

    pub fn delete_dfs(&self, identifier: &str, user_id: String) -> Result<(), Status> {
        let owner_check = self.sess_manager.verify_if_owner(&user_id)?;

        //Removes the memory occupied by this df from memory quota
        let mut memory_quota = self.tracking.memory_quota.write().unwrap();
        let mut dataframe_user = self.tracking.dataframe_user.write().unwrap();

        let dataframe_owner = if owner_check {
            dataframe_user.get(identifier).unwrap()
        } else {
            let dataframe_owner = dataframe_user.get(identifier).unwrap();
            if dataframe_owner == &user_id {
                dataframe_owner
            } else {
                return Err(Status::invalid_argument(
                    "This dataframe does not belong to you.",
                ));
            }
        };

        let mut dfs = self.dataframes.write().unwrap();
        dfs.remove(identifier);

        let path = "data_frames/".to_owned() + identifier + ".json";
        std::fs::remove_file(path).unwrap_or(());

        let (mut consumption, id_sizes) = memory_quota.get(dataframe_owner).unwrap();
        let df_size = id_sizes.get(identifier).unwrap();
        consumption = consumption - df_size;

        let mut id_sizes = id_sizes.to_owned();
        id_sizes.remove(identifier);
        memory_quota.insert(user_id, (consumption, id_sizes));

        dataframe_user.remove(identifier);

        Ok(())
    }
}

fn get_df_header(df: &DataFrame) -> Result<String, Status> {
    serde_json::to_string(&df.schema())
        .map_err(|e| Status::internal(format!("Could not serialize data frame header: {}", e)))
}

#[tonic::async_trait]
impl PolarsService for BastionLabPolars {
    type FetchDataFrameStream = ReceiverStream<Result<FetchChunk, Status>>;

    async fn run_query(
        &self,
        request: Request<Query>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let token = self.sess_manager.get_token(&request)?;

        let composite_plan: CompositePlan = serde_json::from_str(&request.get_ref().composite_plan)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Could not deserialize composite plan: {}{}",
                    e,
                    &request.get_ref().composite_plan
                ))
            })?;
        let user_id = self.sess_manager.get_user_id(token.clone())?;

        let start_time = Instant::now();

        let mut res = composite_plan.run(self, &user_id)?;
        // TODO: this isn't really great.. this does a full serialization under the hood
        let hash = hash_dataset(&mut res.dataframe)
            .map_err(|e| Status::internal(format!("Polars error: {e}")))?;

        let header = get_df_header(&res.dataframe)?;
        let identifier = self.insert_df(res, user_id)?;

        let elapsed = start_time.elapsed();

        telemetry::add_event(
            TelemetryEventProps::RunQuery {
                dataset_name: Some(identifier.clone()),
                dataset_hash: Some(hash),
                time_taken: elapsed.as_millis() as f64,
            },
            Some(self.sess_manager.get_client_info(token)?),
        );

        info!("Succesfully ran query on {}", identifier.clone());

        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        let token = self.sess_manager.get_token(&request)?;
        let user_id = self.sess_manager.get_user_id(token.clone())?;

        let client_info = self.sess_manager.get_client_info(token)?;
        let (df, hash) = unserialize_dataframe(request.into_inner()).await?;
        let header = get_df_header(&df.dataframe)?;
        let identifier = self.insert_df(df, user_id)?;

        let elapsed = start_time.elapsed();
        telemetry::add_event(
            TelemetryEventProps::SendDataFrame {
                dataset_name: Some(identifier.clone()),
                dataset_hash: Some(hash),
                time_taken: elapsed.as_millis() as f64,
            },
            Some(client_info),
        );

        info!(
            "Succesfully sent dataframe {} to server",
            identifier.clone()
        );

        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        let token = self.sess_manager.get_token(&request)?;

        let fut = {
            let df = self.get_df(
                &request.get_ref().identifier,
                Some(self.sess_manager.get_client_info(token)?),
            )?;
            serialize_delayed_dataframe(df)
        };
        Ok(fut.await)
    }

    async fn list_data_frames(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceList>, Status> {
        let token = self.sess_manager.get_token(&request)?;

        let list = self
            .get_headers()?
            .into_iter()
            .map(|(identifier, header)| ReferenceResponse { identifier, header })
            .collect();
        telemetry::add_event(
            TelemetryEventProps::ListDataFrame {},
            Some(self.sess_manager.get_client_info(token)?),
        );
        Ok(Response::new(ReferenceList { list }))
    }

    async fn get_data_frame_header(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let token = self.sess_manager.get_token(&request)?;

        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;
        telemetry::add_event(
            TelemetryEventProps::GetDataFrameHeader {
                dataset_name: Some(identifier.clone()),
            },
            Some(self.sess_manager.get_client_info(token)?),
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn persist_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Empty>, Status> {
        let token = self.sess_manager.get_token(&request)?;

        let identifier = &request.get_ref().identifier;
        self.persist_df(identifier)?;
        telemetry::add_event(
            TelemetryEventProps::SaveDataframe {
                dataset_name: Some(identifier.clone()),
            },
            Some(self.sess_manager.get_client_info(token)?),
        );
        Ok(Response::new(Empty {}))
    }

    async fn delete_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Empty>, Status> {
        let token = self.sess_manager.get_token(&request)?;

        let identifier = &request.get_ref().identifier;
        let user_id = self.sess_manager.get_user_id(token.clone())?;
        self.delete_dfs(identifier, user_id)?;
        telemetry::add_event(
            TelemetryEventProps::DeleteDataframe {
                dataset_name: Some(identifier.clone()),
            },
            Some(self.sess_manager.get_client_info(token)?),
        );

        info!(
            "Succesfully deleted dataframe {} from the server",
            identifier.clone()
        );

        Ok(Response::new(Empty {}))
    }

    async fn split(
        &self,
        request: Request<SplitRequest>,
    ) -> Result<Response<ReferenceList>, Status> {
        #[allow(unused)]
        let (arrays, train_size, test_size, shuffle, random_state) = (
            &request.get_ref().arrays,
            request.get_ref().train_size as f64,
            request.get_ref().test_size as f64,
            request.get_ref().shuffle,
            request.get_ref().random_state,
        );

        let mut out_arrays_store = vec![];
        let mut out_arrays = vec![];
        for array in arrays {
            out_arrays_store.push(self.get_array(&array.identifier)?);
        }

        /*
         - We use StdRng to shuffle indexes of the array.

         -  We then use ArrayBase::select([indices]) to select the respective indices
            along the Axis(0).

         -  The reason why we are not using `rand::sample_axis_using` is that
            for shuffling arrays, if the are multiple, we will want to have the same indexes
            shuffled across.

            This is very important for ML/DL because you wouldn't want to match a different input
            to a different output.
        */
        let mut rng = if let Some(seed) = random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(thread_rng()).unwrap()
        };

        let mut shuffled = false;
        let mut indices = vec![];

        for array in out_arrays_store.iter() {
            if !shuffled {
                indices = (0..array.height()).collect();
                indices.shuffle(&mut rng);
                shuffled = true;
            }
            let array = if shuffle {
                let array = array.shuffle(&indices[..]);
                array
            } else {
                array.clone()
            };

            let (upper, lower) = array.split((train_size, test_size));
            out_arrays.append(&mut vec![
                ReferenceResponse {
                    identifier: self.insert_array(upper),
                    header: String::default(),
                },
                ReferenceResponse {
                    identifier: self.insert_array(lower),
                    header: String::default(),
                },
            ])
        }

        Ok(Response::new(ReferenceList { list: out_arrays }))
    }
}
