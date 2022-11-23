use polars::prelude::*;
use prost::Message;
use serde_json;
use std::{
    collections::HashMap,
    error::Error,
    fmt::Debug,
    fs::{self, File},
    future::Future,
    io::Read,
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    metadata::{KeyRef, MetadataMap},
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
    ChallengeResponse, Empty, FetchChunk, Query, ReferenceList, ReferenceRequest,
    ReferenceResponse, SendChunk, Session,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

mod authentication;
use authentication::*;

mod config;
use config::*;

use ring::rand;

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

#[derive(Debug, Default)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrameArtifact>>>,
    keys: Mutex<KeyManagement>,
    sessions: Arc<RwLock<HashMap<[u8; 32], (SocketAddr, SystemTime, String)>>>,
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

    fn get_df(&self, identifier: &str) -> Result<DelayedDataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        let artifact = dfs.get(identifier).ok_or(Status::not_found(format!(
            "Could not find dataframe: identifier={}",
            identifier
        )))?;
        Ok(match &artifact.fetchable {
            PolicyAction::Accept => {
                let mut df = artifact.dataframe.clone();
                sanitize_df(&mut df, &artifact.blacklist)?;
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
                                "n" => return Err(Status::permission_denied(format!(
                                    "The data owner rejected the fetch operation.
Fetching a dataframe obtained with a non privacy-preserving query requires the approval of the data owner.
This dataframe was obtained in a non privacy-preserving fashion.
Reason: {}",
                                    reason
                                ))),
                                _ => continue,
                            }
                        }
                        Ok({
                            let guard = dfs.read().unwrap();
                            let artifact = guard.get(&identifier).ok_or(Status::not_found(
                                format!("Could not find dataframe: identifier={}", identifier),
                            ))?;
                            let mut df = artifact.dataframe.clone();
                            sanitize_df(&mut df, &artifact.blacklist)?;
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

    fn verify_request(
        &self,
        remote_addr: Option<SocketAddr>,
        metadata: &MetadataMap,
    ) -> Result<(), Status> {
        let mut tokens = self.sessions.write().unwrap();
        if let Some(user_ip) = remote_addr {
            let meta = metadata
                .get_bin("accesstoken-bin")
                .ok_or_else(|| Status::invalid_argument("No accesstoken in request metadata"))?;
            let token = meta
                .to_bytes()
                .map_err(|_| Status::invalid_argument("Could not decode accesstoken"))?;

            if let Some((stored_ip, expiry, _)) = tokens.get(token.as_ref()) {
                let curr_time = SystemTime::now();
                if !verify_ip(&stored_ip, &user_ip) {
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

    fn create_session<T: Message>(&self, request: &Request<T>) -> Result<Session, Status> {
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
                                    let message = get_message(b"create-session", request)?;
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

            sessions.insert(token.clone(), (user_ip, expiry, public_key.clone()));

            Ok(Session {
                token: token.to_vec(),
            })
        } else {
            Err(Status::aborted("Could not fetch IP Address from request"))
        }
    }

    fn refresh_session<T: Message>(&self, request: &Request<T>) -> Result<(), Status> {
        let mut sessions = self.sessions.write().unwrap();
        let meta = request
            .metadata()
            .get_bin("accesstoken-bin")
            .ok_or_else(|| Status::invalid_argument("No accesstoken in request metadata"))?;
        let token = meta
            .to_bytes()
            .map_err(|_| Status::invalid_argument("Could not decode accesstoken"))?;
        let Some((_, expiry, _)) = sessions.get_mut(token.as_ref()) else {
            return Err(Status::aborted("Session not found!"));
        };

        let Some(e) = expiry.checked_add(Duration::from_secs(self.session_expiry)) else {
            return Err(Status::aborted("Malformed session expiry time!"));
        };

        *expiry = e;
        Ok(())
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
        self.verify_request(request.remote_addr(), request.metadata())?;

        let composite_plan: CompositePlan = serde_json::from_str(&request.get_ref().composite_plan)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Could not deserialize composite plan: {}{}",
                    e,
                    &request.get_ref().composite_plan
                ))
            })?;
        let res = composite_plan.run(self)?;

        let header = get_df_header(&res.dataframe)?;
        let identifier = self.insert_df(res);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.verify_request(request.remote_addr(), request.metadata())?;
        let df = df_artifact_from_stream(request.into_inner()).await?;

        let header = get_df_header(&df.dataframe)?;
        let identifier = self.insert_df(df);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        self.verify_request(request.remote_addr(), request.metadata())?;

        let fut = {
            let df = self.get_df(&request.get_ref().identifier)?;
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
        self.verify_request(request.remote_addr(), request.metadata())?;
        let list = self
            .get_headers()?
            .into_iter()
            .map(|(identifier, header)| ReferenceResponse { identifier, header })
            .collect();

        Ok(Response::new(ReferenceList { list }))
    }

    async fn get_data_frame_header(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.verify_request(request.remote_addr(), request.metadata())?;
        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;

        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn create_session(&self, request: Request<Empty>) -> Result<Response<Session>, Status> {
        let session = self.create_session(&request)?;
        Ok(Response::new(session))
    }

    async fn refresh_session(&self, request: Request<Empty>) -> Result<Response<Empty>, Status> {
        self.refresh_session(&request)?;
        Ok(Response::new(Empty {}))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut file = File::open("config.toml")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: BastionLabConfig = toml::from_str(&contents)?;
    let keys = KeyManagement::load_from_dir(config.public_keys_directory()?)?;
    let state = BastionLabState::new(keys, config.session_expiry()?);

    let server_cert = fs::read("tls/host_server.pem")?;
    let server_key = fs::read("tls/host_server.key")?;
    let server_identity = Identity::from_pem(&server_cert, &server_key);

    println!("BastionLab server running...");
    Server::builder()
        .tls_config(ServerTlsConfig::new().identity(server_identity))?
        .add_service(BastionLabServer::new(state))
        .serve(config.client_to_enclave_untrusted_socket()?)
        .await?;
    Ok(())
}
