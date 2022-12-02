use env_logger::Env;
use log::info;
use polars::prelude::*;
use ring::{digest, rand};

use serde_json;
use std::{
    collections::hash_map::DefaultHasher,
    collections::HashMap,
    error::Error,
    fmt::Debug,
    future::Future,
    hash::{Hash, Hasher},
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
    time::{Duration, SystemTime},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    metadata::KeyRef,
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
    ChallengeResponse, ClientInfo, Empty, FetchChunk, Query, ReferenceList, ReferenceRequest,
    ReferenceResponse, SendChunk, SessionInfo,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

pub mod authentication;
use authentication::*;

pub mod config;
use config::*;

mod visitable;

mod telemetry;
use telemetry::TelemetryEventProps;

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
pub struct Session {
    user_ip: SocketAddr,
    expiry: SystemTime,
    public_key: String,
    client_info: ClientInfo,
}

#[derive(Debug, Default)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
    keys: Mutex<KeyManagement>,
    sessions: Arc<RwLock<HashMap<[u8; 32], Session>>>,
    session_expiry: u64,
}

impl BastionLabState {
    fn new(keys: KeyManagement, session_expiry: u64) -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            keys: Mutex::new(keys),
            sessions: Default::default(),
            session_expiry,
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

    fn verify_request<T>(&self, req: &Request<T>) -> Result<(), Status> {
        let remote_addr = &req.remote_addr();
        let token = get_token(req)?;
        let mut tokens = self.sessions.write().unwrap();
        if let Some(recv_ip) = remote_addr {
            if let Some(Session {
                user_ip, expiry, ..
            }) = tokens.get(token.as_ref())
            {
                let curr_time = SystemTime::now();
                if !verify_ip(&user_ip, &recv_ip) {
                    return Err(Status::aborted("Unknown IP Address!"));
                }
                if curr_time.gt(expiry) {
                    tokens.remove(token.as_ref());
                    return Err(Status::aborted("Session Expired"));
                }
            }
        }

        Ok(())
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

    fn new_challenge(&self) -> [u8; 32] {
        let rng = rand::SystemRandom::new();
        loop {
            if let Ok(challenge) = rand::generate(&rng) {
                return challenge.expose();
            }
        }
    }

    fn create_session(&self, request: Request<ClientInfo>) -> Result<SessionInfo, Status> {
        let end = "-bin";
        let pat = "signature-";
        let mut public_key = String::new();
        if let Some(user_ip) = request.remote_addr() {
            for key in request.metadata().keys() {
                match key {
                    KeyRef::Binary(key) => {
                        let key = key.to_string();
                        if let Some(key) = key.strip_suffix(end) {
                            if key.contains(pat) {
                                if let Some(key) = key.split(pat).last() {
                                    let lock = self.keys.lock().unwrap();
                                    let message = get_message(b"create-session", &request)?;
                                    lock.verify_signature(key, &message[..], request.metadata())?;
                                    public_key.push_str(key);
                                } else {
                                    return Err(Status::aborted(
                                        "User signing key not found in request!",
                                    ));
                                }
                            }
                        } else {
                            return Err(Status::aborted("User signing key not found in request!"));
                        }
                    }
                    _ => (),
                }
            }
            let mut sessions = self.sessions.write().unwrap();
            let token = self.new_challenge();
            let Some(expiry) = SystemTime::now().checked_add(Duration::from_secs(self.session_expiry)) else {
                return Err(Status::aborted("Could not create expiry for session"));
            };

            sessions.insert(
                token.clone(),
                Session {
                    user_ip,
                    expiry,
                    public_key,
                    client_info: request.into_inner(),
                },
            );

            Ok(SessionInfo {
                token: token.to_vec(),
            })
        } else {
            Err(Status::aborted("Could not fetch IP Address from request"))
        }
    }

    fn refresh_session<T>(&self, req: &Request<T>) -> Result<(), Status> {
        let token = get_token(req)?;
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(&token[..])
            .ok_or(Status::aborted("Session not found!"))?;

        let e = session
            .expiry
            .checked_add(Duration::from_secs(self.session_expiry))
            .ok_or(Status::aborted("Malformed session expiry time!"))?;

        session.expiry = e;
        Ok(())
    }

    fn get_client_info<T>(&self, req: &Request<T>) -> Result<ClientInfo, Status> {
        let token = get_token(req)?;
        let sessions = self.sessions.write().unwrap();
        let session = sessions
            .get(&token[..])
            .ok_or(Status::aborted("Session not found!"))?;
        Ok(session.client_info.clone())
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
        self.verify_request(&request)?;

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
            Some(self.get_client_info(&request)?),
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        self.verify_request(&request)?;

        let client_info = self.get_client_info(&request)?;
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
        self.verify_request(&request)?;

        let fut = {
            let df = self.get_df(
                &request.get_ref().identifier,
                Some(self.get_client_info(&request)?),
            )?;
            stream_data(df, 32)
        };
        Ok(fut.await)
    }

    async fn get_challenge(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ChallengeResponse>, Status> {
        let challenge = self.new_challenge();
        Ok(Response::new(ChallengeResponse {
            value: challenge.into(),
        }))
    }
    async fn list_data_frames(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceList>, Status> {
        self.verify_request(&request)?;
        let list = self
            .get_headers()?
            .into_iter()
            .map(|(identifier, header)| ReferenceResponse { identifier, header })
            .collect();
        telemetry::add_event(
            TelemetryEventProps::ListDataFrame {},
            Some(self.get_client_info(&request)?),
        );
        Ok(Response::new(ReferenceList { list }))
    }

    async fn get_data_frame_header(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.verify_request(&request)?;
        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;
        telemetry::add_event(
            TelemetryEventProps::GetDataFrameHeader {
                dataset_name: Some(identifier.clone()),
            },
            Some(self.get_client_info(&request)?),
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn create_session(
        &self,
        request: Request<ClientInfo>,
    ) -> Result<Response<SessionInfo>, Status> {
        let session = self.create_session(request)?;
        Ok(Response::new(session))
    }

    async fn refresh_session(&self, request: Request<Empty>) -> Result<Response<Empty>, Status> {
        self.refresh_session(&request)?;
        Ok(Response::new(Empty {}))
    }
}

pub async fn start(
    config: BastionLabConfig,
    keys: KeyManagement,
    server_identity: Identity,
) -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let state = BastionLabState::new(keys, config.session_expiry()?);

    info!("BastionLab server running...");

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
        .tls_config(ServerTlsConfig::new().identity(server_identity))?
        .add_service(BastionLabServer::new(state))
        .serve(config.client_to_enclave_untrusted_socket()?)
        .await?;
    Ok(())
}
