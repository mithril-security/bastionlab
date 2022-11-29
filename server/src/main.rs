use bastionlab_common::prelude::*;
use bastionlab_common::{
    auth::KeyManagement,
    session::{SessionGrpcService, SessionManager},
    session_proto::{self, ClientInfo},
    telemetry::{self, TelemetryEventProps},
};
use bastionlab_torch::torch_proto;
use env_logger::Env;
use log::info;
use polars::prelude::*;
use ring::digest;
use serde_json;
use std::path::Path;
use std::{
    collections::hash_map::DefaultHasher,
    fs::{self, File},
    future::Future,
    io::Read,
    pin::Pin,
    time::Instant,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    transport::{Identity, Server, ServerTlsConfig},
    Request, Response, Status, Streaming,
};
use utils::sanitize_df;
use uuid::Uuid;

pub mod grpc {
    tonic::include_proto!("bastionlab");
}
use grpc::{
    bastion_lab_server::{BastionLab, BastionLabServer},
    Empty, FetchChunk, Query, ReferenceList, ReferenceRequest, ReferenceResponse, SendChunk,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

mod config;
use config::*;

mod visitable;

mod access_control;
use access_control::*;

mod utils;

pub struct DelayedDataFrame {
    future: Pin<Box<dyn Future<Output = Result<DataFrame, Status>> + Send>>,
    needs_approval: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DataFrameArtifact {
    dataframe: DataFrame,
    policy: Policy,
    fetchable: PolicyAction,
    blacklist: Vec<String>,
    query_details: String,
}

impl DataFrameArtifact {
    pub fn new(df: DataFrame, policy: Policy, blacklist: Vec<String>) -> Self {
        DataFrameArtifact {
            dataframe: df,
            policy,
            fetchable: PolicyAction::Reject(String::from(
                "DataFrames uploaded by the Data Owner are protected.",
            )),
            blacklist,
            query_details: String::from("uploaded dataframe"),
        }
    }
}

#[derive(Debug)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
    sess_manager: Arc<SessionManager>,
}

impl BastionLabState {
    fn new(sess_manager: Arc<SessionManager>) -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            sess_manager,
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
        Ok(match &artifact.fetchable {
            PolicyAction::Accept => {
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
                    needs_approval: None,
                }
            }
            PolicyAction::Reject(reason) => {
                let reason = reason.clone();
                DelayedDataFrame {
                    future: Box::pin(async move {
                        Err(Status::permission_denied(format!(
                        "Cannot fetch this DataFrame: operation denied by the data owner's policy
Reason: {}",
                        reason,
                    )))
                    }),
                    needs_approval: None,
                }
            }
            PolicyAction::Approval(reason) => {
                let reason = reason.clone();
                let identifier = String::from(identifier);
                let query_details = artifact.query_details.clone();
                let dfs = Arc::clone(&self.dataframes);
                DelayedDataFrame {
                    needs_approval: Some(reason.clone()),
                    future: Box::pin(async move {
                        println!(
                            "=== A user request has been rejected ===
Reason: {}
Logical plan:
{}",
                            reason, query_details,
                        );

                        loop {
                            let mut ans = String::new();
                            println!("Accept [y] or Reject [n]?");
                            std::io::stdin()
                                .read_line(&mut ans)
                                .expect("Failed to read line");

                            match ans.trim() {
                                "y" => break,
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
}

fn get_df_header(df: &DataFrame) -> Result<String, Status> {
    serde_json::to_string(&df.schema())
        .map_err(|e| Status::internal(format!("Could not serialize data frame header: {}", e)))
}

#[tonic::async_trait]
impl BastionLab for BastionLabState {
    type FetchDataFrameStream = ReceiverStream<Result<FetchChunk, Status>>;

    async fn run_query(
        &self,
        request: Request<Query>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;

        let composite_plan: CompositePlan = serde_json::from_str(&request.get_ref().composite_plan)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Could not deserialize composite plan: {}{}",
                    e,
                    &request.get_ref().composite_plan
                ))
            })?;

        let start_time = Instant::now();

        let res = composite_plan.run(self)?;
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
            Some(self.sess_manager.get_client_info(&request)?),
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        self.sess_manager.verify_request(&request)?;

        let client_info = self.sess_manager.get_client_info(&request)?;
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
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        self.sess_manager.verify_request(&request)?;

        let fut = {
            let df = self.get_df(
                &request.get_ref().identifier,
                Some(self.sess_manager.get_client_info(&request)?),
            )?;
            stream_data(df, 32)
        };
        Ok(fut.await)
    }

    async fn list_data_frames(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceList>, Status> {
        self.sess_manager.verify_request(&request)?;
        let list = self
            .get_headers()?
            .into_iter()
            .map(|(identifier, header)| ReferenceResponse { identifier, header })
            .collect();
        telemetry::add_event(
            TelemetryEventProps::ListDataFrame {},
            Some(self.sess_manager.get_client_info(&request)?),
        );
        Ok(Response::new(ReferenceList { list }))
    }

    async fn get_data_frame_header(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;
        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;
        telemetry::add_event(
            TelemetryEventProps::GetDataFrameHeader {
                dataset_name: Some(identifier.clone()),
            },
            Some(self.sess_manager.get_client_info(&request)?),
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let mut file = File::open("config.toml").context("Opening config.toml file")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .context("Reading the config.toml file")?;

    let config: BastionLabConfig =
        toml::from_str(&contents).context("Parsing the config.toml file")?;
    let keys = KeyManagement::load_from_dir(Path::new(
        &config
            .public_keys_directory()
            .context("Parsing the public_keys_directory config path")?,
    ))
    .context("Loading the stored user keys")?;
    let sess_manager = Arc::new(SessionManager::new(
        keys,
        config
            .session_expiry()
            .context("Parsing the public session_expiry config")?,
    ));

    let server_cert =
        fs::read("tls/host_server.pem").context("Reading the tls/host_server.pem file")?;
    let server_key =
        fs::read("tls/host_server.key").context("Reading the tls/host_server.key file")?;
    let server_identity = Identity::from_pem(&server_cert, &server_key);

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
        telemetry::setup(platform, uid, tee_mode).context("Setting up telemetry")?;
        info!("Telemetry is enabled.")
    } else {
        info!("Telemetry is disabled.")
    }
    telemetry::add_event(TelemetryEventProps::Started {}, None);

    let mut builder = Server::builder()
        .tls_config(ServerTlsConfig::new().identity(server_identity))
        .context("Setting up TLS")?;

    // Session
    let svc = SessionGrpcService::new(sess_manager.clone());
    let builder =
        builder.add_service(session_proto::session_service_server::SessionServiceServer::new(svc));

    // Arrow
    let svc = BastionLabState::new(sess_manager.clone());
    let builder = builder.add_service(BastionLabServer::new(svc));

    // Torch
    let svc = bastionlab_torch::BastionLabTorch::new(sess_manager.clone());
    let builder = builder.add_service(torch_proto::remote_torch_server::RemoteTorchServer::new(
        svc,
    ));

    info!("BastionLab server has been started.");

    // serve!
    builder
        .serve(
            config
                .client_to_enclave_untrusted_socket()
                .context("Parsing the client_to_enclave_untrusted_socket config")?,
        )
        .await?;
    Ok(())
}
