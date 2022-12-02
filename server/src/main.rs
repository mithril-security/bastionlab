use env_logger::Env;
use log::info;
use polars::prelude::*;
use prost::Message;
use ring::{digest, rand};

use serde_json;
use std::{
    collections::HashMap,
    collections::{hash_map::DefaultHasher, HashSet},
    error::Error,
    fmt::Debug,
    fs::{self, File},
    future::Future,
    hash::{Hash, Hasher},
    io::Read,
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
    time::{Duration, SystemTime},
};
use bytes::Bytes;
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

mod authentication;
use authentication::*;

mod config;
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
    client_info: ClientInfo,
}

#[derive(Debug, Default)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
    keys: Mutex<Option<KeyManagement>>,
    sessions: Arc<RwLock<HashMap<[u8; 32], Session>>>,
    session_expiry: u64,
    auth_enabled: bool,
    challenges: Mutex<HashSet<[u8; 32]>>,
}

impl BastionLabState {
    fn new(keys: Option<KeyManagement>, session_expiry: u64) -> Self {
        let auth_enabled = keys.is_some();
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            keys: Mutex::new(keys),
            sessions: Default::default(),
            session_expiry,
            auth_enabled,
            challenges: Default::default(),
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

    fn verify_request<T>(&self, req: &Request<T>, token:Option<Bytes>) -> Result<(), Status> {
        let lock = self.keys.lock().unwrap();
        match *lock {
            Some(_) => {
                match token {
                    Some(token) => {
                        let mut tokens = self.sessions.write().unwrap();
                        let session = tokens.get(token.as_ref()).ok_or(Status::aborted("Session not found!"))?;                        
                        let recv_ip = &req.remote_addr().ok_or(Status::aborted("User IP unavailable"))?;
                        let curr_time = SystemTime::now();

                        if !verify_ip(&session.user_ip, &recv_ip) {
                            return Err(Status::aborted("Unknown IP Address!"));
                        }

                        if curr_time.gt(&session.expiry) {
                            tokens.remove(token.as_ref());
                            return Err(Status::aborted("Session Expired"));
                        }

                    },

                    None => {},
                }
            },

            None => drop(lock),
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
                let challenge: [u8; 32] = challenge.expose();
                let mut lock = self.challenges.lock().unwrap();
                lock.insert(challenge);
                return challenge;
            }
        }
    }
    fn check_challenge<T: Message>(&self, request: &Request<T>) -> Result<(), Status> {
        let mut lock = self.challenges.lock().unwrap();
        if let Some(challenge) = request.metadata().get_bin("challenge-bin") {
            let challenge = challenge.to_bytes().map_err(|_| {
                Status::invalid_argument(format!("Could not decode challenge {:?}", challenge))
            })?;
            let challenge = challenge.as_ref();

            if !lock.remove(challenge) {
                return Err(Status::permission_denied("Challenge not found!"));
            }
        }
        Ok(())
    }

    fn create_session(&self, request: Request<ClientInfo>) -> Result<SessionInfo, Status> {
        self.check_challenge(&request)?;
        let mut sessions = self.sessions.write().unwrap();
        let keys_lock = self.keys.lock().unwrap();
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
                                    if let Some(keys) = &*keys_lock {
                                        let lock = keys;
                                        let message = get_message(b"create-session", &request)?;
                                        lock.verify_signature(
                                            key,
                                            &message[..],
                                            request.metadata(),
                                        )?;
                                        public_key.push_str(key);
                                    }
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
            let (token, expiry) = if !self.auth_enabled {
                ([0u8; 32], SystemTime::now())
            } else {
                let expiry =
                    match SystemTime::now().checked_add(Duration::from_secs(self.session_expiry)) {
                        Some(v) => v,
                        None => SystemTime::now(),
                    };
                (self.new_challenge(), expiry)
            };

            sessions.insert(
                token.clone(),
                Session {
                    user_ip,
                    expiry,
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
        if let Some(token) = get_token(req, self.auth_enabled)? {
            let mut sessions = self.sessions.write().unwrap();
            let session = sessions
                .get_mut(&token[..])
                .ok_or(Status::aborted("Session not found!"))?;

            let e = session
                .expiry
                .checked_add(Duration::from_secs(self.session_expiry))
                .ok_or(Status::aborted("Malformed session expiry time!"))?;

            session.expiry = e;
        }
        Ok(())
    }

    fn get_client_info(&self, token: Option<Bytes>) -> Result<ClientInfo, Status> {
        let sessions = self.sessions.write().unwrap();
        let token = match &token {
            Some(v) => &v[..],
            None => &[0u8; 32],
        };
        let session = sessions
            .get(token.as_ref())
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
        
        let token = get_token(&request, self.auth_enabled)?;
        self.verify_request(&request,token.clone())?;

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
            Some(self.get_client_info(token)?),
        );
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        let token = get_token(&request, self.auth_enabled)?;
        self.verify_request(&request,token.clone())?;

        let client_info = self.get_client_info(token)?;
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
        let token = get_token(&request, self.auth_enabled)?;
        self.verify_request(&request,token.clone())?;

        let fut = {
            let df = self.get_df(
                &request.get_ref().identifier,
                Some(self.get_client_info(token)?),
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
        let token = get_token(&request, self.auth_enabled)?;
        self.verify_request(&request,token.clone())?;

        let list = self
            .get_headers()?
            .into_iter()
            .map(|(identifier, header)| ReferenceResponse { identifier, header })
            .collect();
        telemetry::add_event(
            TelemetryEventProps::ListDataFrame {},
            Some(self.get_client_info(token)?),
        );
        Ok(Response::new(ReferenceList { list }))
    }

    async fn get_data_frame_header(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let token = get_token(&request, self.auth_enabled)?;
        self.verify_request(&request,token.clone())?;

        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;
        telemetry::add_event(
            TelemetryEventProps::GetDataFrameHeader {
                dataset_name: Some(identifier.clone()),
            },
            Some(self.get_client_info(token)?),
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let mut file = File::open("config.toml")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    println!("BastionLab server running...");

    let config: BastionLabConfig = toml::from_str(&contents)?;

    let disable_authentication = !std::env::var("DISABLE_AUTHENTICATION").is_err();
    
    let keys = match KeyManagement::load_from_dir(config.public_keys_directory()?) {
        Ok(keys) => {
            if !disable_authentication {
                info!("Authentication is enabled.");
                Some(keys)
            }
            else {
                info!("Authentication is disabled.");
                None
            }
        }
        Err(e) =>  {
            println!("Exiting due to an error reading keys. {}", e.message());
            //Temp fix to exit early, returning an error seems to break the "?" handlers above.
            return Ok(())
        }
    };

    let state = BastionLabState::new(keys, config.session_expiry()?);

    let server_cert = fs::read("tls/host_server.pem")?;
    let server_key = fs::read("tls/host_server.key")?;
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
