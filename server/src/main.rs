use polars::prelude::*;
use serde_json;
use std::{
    collections::HashMap,
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
    Chunk, Empty, Query, ReferenceRequest, ReferenceResponse, References, TrainingRequest,
};

// Training routines
mod operations;
use operations::*;

mod trainer;
use trainer::*;

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

#[derive(Debug, Default)]
pub struct BastionLabState {
    // queries: Arc<Vec<String>>,
    dataframes: Arc<RwLock<HashMap<String, DataFrame>>>,
    models: Arc<RwLock<HashMap<String, SupportedModels>>>,
}

impl BastionLabState {
    fn new() -> Self {
        Self {
            // queries: Arc::new(Vec::new()),
            dataframes: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
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

    // fn get_dfs(&self, identifiers: &[String]) -> Result<VecDeque<DataFrame>, Status> {
    //     let dfs = self.dataframes.read().unwrap();
    //     let mut res = VecDeque::with_capacity(identifiers.len());
    //     for identifier in identifiers.iter() {
    //         res.push_back(dfs.get(identifier).ok_or(Status::not_found(format!("Could not find dataframe: identifier={}", identifier)))?.clone());
    //     }
    //     Ok(res)
    // }

    fn insert_df(&self, df: DataFrame) -> String {
        let mut dfs = self.dataframes.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        dfs.insert(identifier.clone(), df);
        identifier
    }

    fn insert_model(&self, model: SupportedModels) -> String {
        let mut models = self.models.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        models.insert(identifier.clone(), model);
        identifier
    }

    // fn get_model(&self, identifier: &str) -> Result<SupportedModels, Status> {
    //     let models = self.models.read().unwrap();
    //     let model = models.get(identifier).unwrap();
    //     Ok(model.clone())
    // }
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
        let dfs = self.dataframes.read().unwrap();
        let list = dfs
            .iter()
            .map(|(k, v)| ReferenceResponse {
                identifier: k.clone(),
                header: format!("{:?}", v.schema()),
            })
            .collect::<Vec<ReferenceResponse>>();
        println!("{:?}", list);
        Ok(Response::new(References { list }))
    }

    async fn train(
        &self,
        request: Request<TrainingRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let (records, target, ratio, trainer): (String, String, f32, &str) = (
            request.get_ref().records.clone(),
            request.get_ref().target.clone(),
            request.get_ref().ratio,
            request.get_ref().trainer.as_str(),
        );

        let dfs = self.dataframes.read().unwrap();
        let (records, target) = {
            let records = dfs.get(&records).unwrap();
            let target = dfs.get(&target).unwrap();
            (records, target)
        };

        let trainer = match trainer {
            "GaussianNaiveBayes" => Models::GaussianNaiveBayes,
            "ElasticNet" => Models::ElasticNet,
            _ => {
                return Err(Status::aborted(format!(
                    "Unsupported trainer type: {:?}!",
                    trainer
                )));
            }
        };
        let model = to_polars_error(send_to_trainer(
            records.clone(),
            target.clone(),
            ratio,
            trainer,
        ))
        .map_err(|e| Status::aborted(e.to_string()))?;
        let identifier = self.insert_model(model);
        Ok(Response::new(ReferenceResponse {
            identifier,
            header: String::default(),
        }))
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let state = BastionLabState::new();
    let addr = "[::1]:50056".parse()?;
    println!("BastionLab server running...");
    // println!("{:?}", serde_json::from_str::<CompositePlan>("[{\"EntryPointPlanSegment\":\"1da61d9a-c8a8-4e8e-baec-b132db9009d9\"},{\"EntryPointPlanSegment\":\"1da61d9a-c8a8-4e8e-baec-b132db9009d9\"}]").unwrap());
    Server::builder()
        .add_service(BastionLabServer::new(state))
        .serve(addr)
        .await?;
    Ok(())
}
