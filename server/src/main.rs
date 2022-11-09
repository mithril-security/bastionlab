use polars::prelude::*;
use serde_json;
use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    fmt::Debug,
    sync::{Arc, RwLock},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod grpc {
    tonic::include_proto!("bastionlab");
}
use grpc::{
    bastion_lab_server::{BastionLab, BastionLabServer},
    Chunk, Empty, Query, ReferenceRequest, ReferenceResponse, References,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

#[derive(Debug, Default)]
pub struct BastionLabState {
    // queries: Arc<Vec<String>>,
    dataframes: Arc<RwLock<HashMap<String, DataFrame>>>,
}

impl BastionLabState {
    fn new() -> Self {
        Self {
            // queries: Arc::new(Vec::new()),
            dataframes: Arc::new(RwLock::new(HashMap::new())),
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

    fn get_dfs(&self) -> Result<VecDeque<(String, DataFrame)>, Status> {
        let identifiers = {
            let identifiers = self.dataframes.read().unwrap();
            let identifiers: Vec<String> = identifiers.keys().map(|v| v.clone()).collect();
            identifiers
        };
        let dfs = self.dataframes.read().unwrap();
        let mut res = VecDeque::with_capacity(identifiers.len());
        for identifier in identifiers.iter() {
            res.push_back((
                identifier.clone(),
                dfs.get(identifier)
                    .ok_or(Status::not_found(format!(
                        "Could not find dataframe: identifier={}",
                        identifier
                    )))?
                    .clone(),
            ));
        }
        Ok(res)
    }

    fn insert_df(&self, df: DataFrame) -> String {
        let mut dfs = self.dataframes.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        dfs.insert(identifier.clone(), df);
        identifier
    }
}

#[tonic::async_trait]
impl BastionLab for BastionLabState {
    type FetchDataFrameStream = ReceiverStream<Result<Chunk, Status>>;

    async fn run_query(
        &self,
        request: Request<Query>,
    ) -> Result<Response<ReferenceResponse>, Status> {
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

        let header = serde_json::to_string(&res.schema()).map_err(|e| {
            Status::internal(format!(
                "Could not serialize result data frame header: {}",
                e
            ))
        })?;
        let identifier = self.insert_df(res);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let df = df_from_stream(request.into_inner()).await?;

        let header = serde_json::to_string(&df.schema())
            .map_err(|e| Status::internal(format!("Could not serialize header: {}", e)))?;
        let identifier = self.insert_df(df);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        let df = self.get_df(&request.get_ref().identifier)?;

        Ok(stream_data(df, 32).await)
    }

    async fn available_datasets(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<References>, Status> {
        let dfs = self.get_dfs()?;
        let ids: Vec<ReferenceResponse> = dfs
            .iter()
            .map(|(k, v)| {
                let header = serde_json::to_string(&v.schema())
                    .map_err(|e| {
                        Status::internal(format!(
                            "Could not serialize result data frame header: {}",
                            e
                        ))
                    })
                    .unwrap();
                ReferenceResponse {
                    identifier: k.clone(),
                    header,
                }
            })
            .collect();

        Ok(Response::new(References { list: ids }))
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let state = BastionLabState::new();
    let addr = "0.0.0.0:50056".parse()?;
    println!("BastionLab server running...");
    // println!("{:?}", serde_json::from_str::<CompositePlan>("[{\"EntryPointPlanSegment\":\"1da61d9a-c8a8-4e8e-baec-b132db9009d9\"},{\"EntryPointPlanSegment\":\"1da61d9a-c8a8-4e8e-baec-b132db9009d9\"}]").unwrap());
    Server::builder()
        .add_service(BastionLabServer::new(state))
        .serve(addr)
        .await?;
    Ok(())
}
