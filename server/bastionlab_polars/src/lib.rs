use bastionlab_common::prelude::*;
use bastionlab_common::{
    session::SessionManager,
    session_proto::ClientInfo,
    telemetry::{self, TelemetryEventProps},
};
use UnsafeAction::Log;
use chrono::{Local, DateTime};
use polars::prelude::*;
use ring::digest;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{create_dir, read_dir, OpenOptions};
use std::io::{Error, ErrorKind};
use std::{future::Future, pin::Pin, time::Instant};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use utils::sanitize_df;
use uuid::Uuid;
use crate::Rule::True;
const LOG_DF: &str = "00000000-0000-0000-0000-000000000000";

pub mod polars_proto {
    tonic::include_proto!("bastionlab_polars");
}
use polars_proto::{
    polars_service_server::PolarsService, Empty, FetchChunk, Query, ReferenceList,
    ReferenceRequest, ReferenceResponse, SendChunk,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

mod visitable;

mod access_control;
use access_control::*;

mod utils;

pub enum FetchStatus {
    Ok,
    Pending(String),
    Warning(String),
}

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
}

#[derive(Debug)]
pub struct BastionLabPolars {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
    sess_manager: Arc<SessionManager>,
}

impl BastionLabPolars {
    pub fn new(sess_manager: Arc<SessionManager>) -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            sess_manager,
        }
    }

    fn build_log(
        self: &BastionLabPolars,
        user_hash: String,
        op_type: &str,
        inputs: String,
        output: String,
        cp: String,
        pv: String,
    ) -> Result<(), Status>
    {
        let dt: DateTime<Local> = Local::now();
        let new_row = df! [
        "User"              => [user_hash],
        "Time"              => [dt.to_string()],
        "Type"              => [op_type],
        "Inputs"            => [inputs.clone()],
        "Output"            => [output],
        "CompositePlan"     => [cp],
        "PolicyViolation"   => [pv],
        ].unwrap_or(DataFrame::default());
        if (inputs != LOG_DF) & (new_row != DataFrame::default()) {
            return self.update_log(new_row);
        }
        return Ok(());    
    }

    fn update_log(
        self: &BastionLabPolars,
        new_row: DataFrame,
        ) -> Result<(), Status> {
        {
            let mut dfs = self.dataframes.write().unwrap(); 
            let artifact = dfs.get_mut(LOG_DF);
            match artifact{
                Some(artifact) => { //log dataframe exists, append new_row and save
                    artifact.dataframe = artifact.dataframe.vstack(&new_row).unwrap_or(artifact.dataframe.clone());
                }
                None => {
                    let pol = Policy{
                    safe_zone: True,
                    unsafe_handling: Log,
                    savable: true,
                    };
                    let df = DataFrameArtifact{
                        dataframe: new_row,
                        policy: pol,
                        fetchable: VerificationResult::Safe,
                        blacklist: Vec::new(),
                        query_details: String::from("uploaded dataframe"),
                    };
                    dfs.insert(LOG_DF.to_string(), df);
                }
            }
        }
        return self.persist_df(LOG_DF); 
    }

    fn log_get_inputs(
        self: &BastionLabPolars,
        composite_plan: &CompositePlan,
    ) -> String {
        let mut log_inputs = Vec::new();
            for segment in composite_plan.iter() {
                match segment {
                    CompositePlanSegment::EntryPointPlanSegment(identifier) => {
                        log_inputs.push(identifier.clone());
                    }
                    _ => {}
                }
            }
            let mut inputs = String::from("");
            for i in &log_inputs{
                let mut _x = false;
                if _x == true{
                    inputs.push_str(", ");
                }
                inputs.push_str(i);
                _x = true;
            }
        return inputs;
    }

    fn log_get_pol(
        self: &BastionLabPolars,
        id: &str,
    ) -> String {
        let dfs = self.dataframes.read().unwrap();
        let artifact = dfs.get(id);
        match artifact{
            Some(artifact) => { //log dataframe exists, append new_row and save
                let mut pol = "None";
                if let VerificationResult::Unsafe { action, reason } = &artifact.fetchable {
                    pol = reason;
                    let _action = action;
                }
                return pol.to_string();
            }
            None => {
                return "Error retrieving policy".to_string();
            }
        } 
    }

    fn get_df(
        &self,
        identifier: &str,
        client_info: Option<ClientInfo>,
    ) -> Result<DelayedDataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        let artifact = dfs.get(identifier).ok_or(Status::not_found(format!(
            "Could not find dataframe: identifier={}",
            identifier
        )))?;
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
                            let artifact = guard.get(&identifier).ok_or(Status::not_found(
                                format!("Could not find dataframe: identifier={}", identifier),
                            ))?;
                            let mut df = artifact.dataframe.clone();
                            sanitize_df(&mut df, &artifact.blacklist);
                            df
                        })
                    }),
                }
            }
        })
    }

    fn get_df_unchecked(&self, identifier: &str) -> Result<DataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        Ok(dfs
            .get(identifier)
            .ok_or(Status::not_found(format!(
                "Could not find dataframe: identifier={}",
                identifier
            )))?
            .dataframe
            .clone())
    }

    fn with_df_artifact_ref<T>(
        &self,
        identifier: &str,
        mut f: impl FnMut(&DataFrameArtifact) -> T,
    ) -> Result<T, Status> {
        let dfs = self.dataframes.read().unwrap();
        Ok(f(dfs.get(identifier).ok_or(Status::not_found(format!(
            "Could not find dataframe: identifier={}",
            identifier
        )))?))
    }

    fn get_header(&self, identifier: &str) -> Result<String, Status> {
        Ok(get_df_header(
            &self
                .dataframes
                .read()
                .unwrap()
                .get(identifier)
                .ok_or(Status::not_found(format!(
                    "Could not find dataframe: identifier={}",
                    identifier
                )))?
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

    fn insert_df(&self, df: DataFrameArtifact) -> String {
        let mut dfs = self.dataframes.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        dfs.insert(identifier.clone(), df);
        identifier
    }

    fn persist_df(&self, identifier: &str) -> Result<(), Status> {
        let dataframes = self
            .dataframes
            .read()
            .map_err(|_| Status::internal("Unable to read dataframes!"))?;

        let df_artifact = dataframes
            .get(identifier)
            .ok_or("")
            .map_err(|_| Status::not_found("Unable to find dataframe!"))?;

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
        let token = self.sess_manager.verify_request(&request)?;

        let user_id = self.sess_manager.get_user_id(token.clone())?;

        let composite_plan: CompositePlan = serde_json::from_str(&request.get_ref().composite_plan)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Could not deserialize composite plan: {}{}",
                    e,
                    &request.get_ref().composite_plan
                ))
            })?;

        //For log: get idenitifers and put them into inputs string
        let inputs = self.log_get_inputs(&composite_plan);

        let start_time = Instant::now();
        let res = composite_plan.run(self, &user_id)?;
        let dataframe_bytes: Vec<u8> =
            df_to_bytes(&res.dataframe)
                .iter_mut()
                .fold(Vec::new(), |mut acc, x| {
                    acc.append(x);
                    acc
                }); // Not efficient fix this

        let header = get_df_header(&res.dataframe)?;
        let identifier = self.insert_df(res);

        let elapsed = start_time.elapsed();
        let hash = hex::encode(digest::digest(&digest::SHA256, &dataframe_bytes).as_ref());

        telemetry::add_event(
            TelemetryEventProps::RunQuery {
                dataset_name: Some(identifier.clone()),
                dataset_hash: Some(hash),
                time_taken: elapsed.as_millis() as f64,
            },
            Some(self.sess_manager.get_client_info(token)?),
        );
        info!("Succesfully ran query on {}", identifier.clone());
        //update log
        if !&inputs.clone().starts_with(LOG_DF) {
            let ret = self.build_log(user_id.clone(), "Run", inputs.clone(), identifier.clone(), request.get_ref().composite_plan.clone(), "n/a".to_string());
            match ret {
                Ok(_) => {}
                Err(_err) => {
                    return Err(Status::unknown("Unable to save changed to security log"));
                }
            }
        }
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        let token = self.sess_manager.verify_request(&request)?;

        let client_info = self.sess_manager.get_client_info(token.clone())?;
        let df = df_artifact_from_stream(request.into_inner()).await?;
        let dataframe_bytes: Vec<u8> =
            df_to_bytes(&df.dataframe)
                .iter_mut()
                .fold(Vec::new(), |mut acc, x| {
                    acc.append(x);
                    acc
                }); // Not efficient fix this
        let header = get_df_header(&df.dataframe)?;
        let identifier = self.insert_df(df);

        let elapsed = start_time.elapsed();
        let hash = hex::encode(digest::digest(&digest::SHA256, &dataframe_bytes).as_ref());
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
        let user_hash= self.sess_manager.get_user_id(token.clone())?;
        let ret = self.build_log(user_hash, &"Send".to_string(),  "n/a".to_string(), identifier.clone(), "n/a".to_string(), "n/a".to_string()); 
        match ret {
            Ok(_) => {}
            Err(_err) => {
                return Err(Status::unknown("Unable to save changes to security log"));
            }
        }
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        let token = self.sess_manager.verify_request(&request)?;

        let fut = {
            let df = self.get_df(
                &request.get_ref().identifier,
                Some(self.sess_manager.get_client_info(token.clone())?),
            )?;
            stream_data(df, 32)
        };
        let ret = fut.await;

        //logging
        let user_hash= self.sess_manager.get_user_id(token.clone())?;
        let id = &request.get_ref().identifier[..];
        let pol = self.log_get_pol(id);
        let ret2 = self.build_log(user_hash, "Fetch",  String::from(id), "n/a".to_string(), "n/a".to_string(), pol);
        match ret2 {
            Ok(_) => {}
            Err(_err) => {
                return Err(Status::unknown("Unable to save changes to security log"));
            }
        }
        Ok(ret)
    }

    async fn list_data_frames(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceList>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
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
        let token = self.sess_manager.verify_request(&request)?;
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
        let token = self.sess_manager.verify_request(&request)?;
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
}
