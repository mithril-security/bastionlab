use polars::prelude::*;
use prost::Message;
use serde_json;
use std::{
    collections::HashMap,
    error::Error,
    fs::{self, File},
    io::Read,
    net::SocketAddr,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, SystemTime},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    metadata::KeyRef,
    transport::{Identity, Server, ServerTlsConfig},
    Request, Response, Status, Streaming,
};
use uuid::Uuid;

pub mod grpc {
    tonic::include_proto!("bastionlab");
}
use grpc::{
    bastion_lab_server::{BastionLab, BastionLabServer},
    ChallengeResponse, Chunk, Empty, Query, ReferenceList, ReferenceRequest, ReferenceResponse,
    Session,
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

/// Expiry time for session tokens.
const EXPIRY: u64 = 25 * 60;

#[derive(Debug, Default)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrame>>>,
    keys: Mutex<KeyManagement>,
    sessions: Arc<RwLock<HashMap<[u8; 32], (SocketAddr, SystemTime, String)>>>,
}

impl BastionLabState {
    fn new(keys: KeyManagement) -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            keys: Mutex::new(keys),
            sessions: Default::default(),
        }
    }

    fn get_df(&self, identifier: &str) -> Result<DataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        Ok(dfs
            .get(identifier)
            .ok_or(Status::not_found(format!(
                "Could not find dataframe: identifier={}",
                identifier
            )))?
            .clone())
    }

    fn verify_request<T: Message>(&self, request: &Request<T>) -> Result<(), Status> {
        let mut tokens = self.sessions.write().unwrap();
        if let Some(user_ip) = request.remote_addr() {
            let meta = request
                .metadata()
                .get_bin("accesstoken-bin")
                .ok_or_else(|| Status::invalid_argument("No accesstoken in request metadata"))?;
            let token = meta
                .to_bytes()
                .map_err(|_| Status::invalid_argument("Could not decode accesstoken"))?;

            if let Some((stored_ip, expiry, pub_key)) = tokens.get(token.as_ref()) {
                let curr_time = SystemTime::now();
                pub_key.clone().truncate(16);
                println!("{:?} with {:?} issued request!", pub_key, user_ip);
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
            self.dataframes
                .read()
                .unwrap()
                .get(identifier)
                .ok_or(Status::not_found(format!(
                    "Could not find dataframe: identifier={}",
                    identifier
                )))?,
        )?)
    }

    fn get_headers(&self) -> Result<Vec<(String, String)>, Status> {
        let dataframes = self.dataframes.read().unwrap();
        let mut res = Vec::with_capacity(dataframes.len());
        for (k, v) in dataframes.iter() {
            let header = get_df_header(v)?;
            res.push((k.clone(), header));
        }
        Ok(res)
    }

    fn insert_df(&self, df: DataFrame) -> String {
        let mut dfs = self.dataframes.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        dfs.insert(identifier.clone(), df);
        identifier
    }

    fn new_challenge(&self) -> [u8; 32] {
        let rng = rand::SystemRandom::new();
        loop {
            let challenge: [u8; 32] = rand::generate(&rng)
                .expect("Could not generate random value")
                .expose();
            return challenge;
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
                                    Err(Status::aborted("User signing key not found in request!"))?
                                }
                            }
                        } else {
                            Err(Status::aborted("User signing key not found in request!"))?
                        }
                    }
                    _ => (),
                }
            }
            let mut sessions = self.sessions.write().unwrap();
            let token = self.new_challenge();
            let Some(expiry) = SystemTime::now().checked_add(Duration::from_secs(EXPIRY)) else {
                return Err(Status::aborted("Could not create expiry for session"))?;
            };

            sessions.insert(token.clone(), (user_ip, expiry, public_key.clone()));
            public_key.truncate(16);
            println!("{:?} with IP {} created a session", public_key, user_ip,);
            return Ok(Session {
                token: token.to_vec(),
            });
        } else {
            return Err(Status::aborted("Could not fetch IP Address from request"))?;
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
            return Err(Status::aborted("Session not found!"))?;
        };

        let Some(e) = expiry.checked_add(Duration::from_secs(EXPIRY)) else {
            return Err(Status::aborted("Malformed session expiry time!"))?;
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
    type FetchDataFrameStream = ReceiverStream<Result<Chunk, Status>>;

    async fn run_query(
        &self,
        request: Request<Query>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.verify_request(&request)?;

        // let input_dfs = self.get_dfs(&request.get_ref().identifiers)?;
        println!("{:?}", request);
        println!("{}", &request.get_ref().composite_plan);
        let composite_plan: CompositePlan = serde_json::from_str(&request.get_ref().composite_plan)
            .map_err(|e| {
                Status::invalid_argument(format!(
                    "Could not deserialize composite plan: {}{}",
                    e,
                    &request.get_ref().composite_plan
                ))
            })?;
        let res = composite_plan.run(self)?;

        let header = get_df_header(&res)?;
        let identifier = self.insert_df(res);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let df = df_from_stream(request.into_inner()).await?;

        let header = get_df_header(&df)?;
        let identifier = self.insert_df(df);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        self.verify_request(&request)?;
        let df = self.get_df(&request.get_ref().identifier)?;

        Ok(stream_data(df, 32).await)
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
        _request: Request<Empty>,
    ) -> Result<Response<ReferenceList>, Status> {
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
        self.verify_request(&request)?;
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
    let state = BastionLabState::new(keys);

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
