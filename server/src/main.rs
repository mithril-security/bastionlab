use env_logger::Env;
use log::info;
use polars::prelude::*;
use ring::digest;
use serde_json;
use std::{
    collections::hash_map::DefaultHasher,
    collections::HashMap,
    error::Error,
    fmt::Debug,
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
    time::Instant,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod grpc {
    tonic::include_proto!("bastionlab");
}
use grpc::{
    bastion_lab_server::{BastionLab, BastionLabServer},
    Chunk, ListDataFramesRequest, Query, ReferenceList, ReferenceRequest, ReferenceResponse,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

mod telemetry;
mod visitable;
use telemetry::TelemetryEventProps;

#[derive(Debug, Clone)]
pub struct DataFrameArtifact {
    dataframe: DataFrame,
    fetchable: bool,
    query_details: String,
}

impl DataFrameArtifact {
    pub fn new(df: DataFrame) -> Self {
        DataFrameArtifact {
            dataframe: df,
            fetchable: false,
            query_details: String::from("uploaded dataframe"),
        }
    }
}

#[derive(Debug, Default)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
}

impl BastionLabState {
    fn new() -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn get_df(&self, identifier: &str) -> Result<DataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        let artifact = dfs.get(identifier).ok_or(Status::not_found(format!(
            "Could not find dataframe: identifier={}",
            identifier
        )))?;
        if !artifact.fetchable {
            println!(
                "=== A user request has been rejected ===
        Reason: Cannot fetch non aggregated results with at least {} samples per group.
        Logical plan:
        {}",
                10, artifact.query_details,
            );

            loop {
                let mut ans = String::new();
                println!("Accept [y] or Reject [n]?");
                std::io::stdin()
                    .read_line(&mut ans)
                    .expect("Failed to read line");

                match ans.trim() {
                    "y" => break,
                    "n" => return Err(Status::invalid_argument(format!(
                        "The data owner rejected the fetch operation.
        Fetching a dataframe obtained with a non privacy-preserving query requires the approval of the data owner.
        This dataframe was obtained in a non privacy-preserving fashion as it does not aggregate results with at least {} samples per group.",
                        10
                    ))),
                    _ => continue,
                }
            }
        }
        Ok(artifact.dataframe.clone())
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
}

fn get_df_header(df: &DataFrame) -> Result<String, Status> {
    serde_json::to_string(&df.schema())
        .map_err(|e| Status::internal(format!("Could not serialize data frame header: {}", e)))
}

#[tonic::async_trait]
impl BastionLab for BastionLabState {
    type FetchDataFrameStream = ReceiverStream<Result<Chunk, Status>>;

    async fn run_query(
        &self,
        request: Request<Query>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let composite_plan: CompositePlan = serde_json::from_str(&request.get_ref().composite_plan)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Could not deserialize composite plan: {}{}",
                    e,
                    &request.get_ref().composite_plan
                ))
            })?;

        let client_info = request.get_ref().client_info.clone();
        let start_time = Instant::now();

        let res = composite_plan.run(self)?;
        let dataframe_bytes: Vec<u8> = ref_df_to_bytes(&res.dataframe);

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
            client_info,
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();
        let (df, client_info) = df_from_stream(request.into_inner()).await?;
        let dataframe_bytes: Vec<u8> = ref_df_to_bytes(&df);

        let header = get_df_header(&df)?;

        let identifier = self.insert_df(DataFrameArtifact::new(df));
        let elapsed = start_time.elapsed();
        let hash = hex::encode(digest::digest(&digest::SHA256, &dataframe_bytes).as_ref());
        telemetry::add_event(
            TelemetryEventProps::SendDataFrame {
                dataset_name: Some(identifier.clone()),
                dataset_hash: Some(hash),
                time_taken: elapsed.as_millis() as f64,
            },
            client_info,
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        let client_info = request.get_ref().client_info.clone();
        let df_res = self.get_df(&request.get_ref().identifier);
        match df_res {
            Ok(df) => {
                telemetry::add_event(
                    TelemetryEventProps::FetchDataFrame {
                        dataset_name: Some(request.get_ref().identifier.clone()),
                        request_accepted: true,
                    },
                    client_info,
                );
                return Ok(stream_data(df, 32).await);
            }
            Err(err_status) => {
                telemetry::add_event(
                    TelemetryEventProps::FetchDataFrame {
                        dataset_name: Some(request.get_ref().identifier.clone()),
                        request_accepted: false,
                    },
                    client_info,
                );
                return Err(err_status);
            }
        };
    }

    async fn list_data_frames(
        &self,
        request: Request<ListDataFramesRequest>,
    ) -> Result<Response<ReferenceList>, Status> {
        let client_info = request.get_ref().client_info.clone();
        let list = self
            .get_headers()?
            .into_iter()
            .map(|(identifier, header)| ReferenceResponse { identifier, header })
            .collect();
        telemetry::add_event(
            TelemetryEventProps::ListDataFrame {},
            client_info,
        );
        Ok(Response::new(ReferenceList { list }))
    }

    async fn get_data_frame_header(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let client_info = request.get_ref().client_info.clone();
        let requested_identifier = &request.get_ref().identifier;
        let identifier = String::from(requested_identifier);
        let header = self.get_header(&identifier)?;
        telemetry::add_event(
            TelemetryEventProps::GetDataFrameHeader {
                dataset_name: Some(requested_identifier.clone()),
            },
            client_info,
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let state = BastionLabState::new();
    let addr = "0.0.0.0:50056".parse()?;
    println!("BastionLab server running...");

    //TODO: Change it when specifying the TEE will be available
    let tee_mode = String::from("None");
    let platform: String = String::from(format!("{} - TEE Mode: {}", whoami::platform(), tee_mode));
    let uid: String = {
        let mut hasher = DefaultHasher::new();
        whoami::username().hash(&mut hasher);
        whoami::hostname().hash(&mut hasher);
        platform.hash(&mut hasher);
        String::from(format!("{:X}", hasher.finish()))
    };

    if std::env::var("BASTIONLAB_DISABLE_TELEMETRY").is_err() {
        telemetry::setup(platform, uid, tee_mode)?;
        info!("Telemetry is enabled.")
    } else {
        info!("Telemetry is disabled.")
    }
    telemetry::add_event(TelemetryEventProps::Started {}, None);

    Server::builder()
        .add_service(BastionLabServer::new(state))
        .serve(addr)
        .await?;
    Ok(())
}
