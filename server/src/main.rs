use polars::prelude::*;
use serde_json;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    sync::{Arc, Mutex, RwLock},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{metadata::KeyRef, transport::Server, Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod grpc {
    tonic::include_proto!("bastionlab");
}
use grpc::{
    bastion_lab_server::{BastionLab, BastionLabServer},
    ChallengeResponse, Chunk, Empty, Query, ReferenceList, ReferenceRequest, ReferenceResponse,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

mod access_control;
use access_control::*;

use ring::rand;
#[derive(Debug, Default)]
pub struct BastionLabState {
    dataframes: Arc<RwLock<HashMap<String, DataFrame>>>,
    keys: Mutex<KeyManagement>,
    challenges: Mutex<HashSet<[u8; 32]>>,
}

impl BastionLabState {
    fn new(keys: KeyManagement) -> Self {
        Self {
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            keys: Mutex::new(keys),
            challenges: Default::default(),
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

    fn verify_request<T>(&self, request: &Request<T>) -> Result<(), Status> {
        let pat = "signing-key-";
        let end = "-bin";
        for key in request.metadata().keys() {
            match key {
                KeyRef::Binary(key) => {
                    let key = key.to_string();
                    if let Some(key) = key.strip_suffix(end) {
                        if key.contains(pat) {
                            if let Some(key) = key.split(pat).last() {
                                let lock = self.keys.lock().unwrap();
                                lock.verify_key(key)?;
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
    fn check_challenge<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if let Some(meta) = request.metadata().get_bin("challenge-bin") {
            let challenge = meta
                .to_bytes()
                .map_err(|_| Status::invalid_argument("Could not decode challenge"))?;
            let mut lock = self.challenges.lock().unwrap();
            if !lock.remove(challenge.as_ref()) {
                Err(Status::permission_denied("Invalid or reused challenge"))?
            }
        }
        Ok(())
    }

    fn new_challenge(&self) -> [u8; 32] {
        let rng = rand::SystemRandom::new();
        loop {
            let challenge: [u8; 32] = rand::generate(&rng)
                .expect("Could not generate random value")
                .expose();
            if self.challenges.lock().unwrap().insert(challenge) {
                return challenge;
            }
        }
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
        self.check_challenge(&request)?;
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
        self.check_challenge(&request)?;
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
        self.check_challenge(&request)?;
        self.verify_request(&request)?;
        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;

        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let keys = KeyManagement::load_from_dir("./keys".to_string())?;
    let state = BastionLabState::new(keys);
    let addr = "0.0.0.0:50056".parse()?;
    println!("BastionLab server running...");
    Server::builder()
        .add_service(BastionLabServer::new(state))
        .serve(addr)
        .await?;
    Ok(())
}
