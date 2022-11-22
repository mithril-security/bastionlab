use polars::prelude::*;
use serde_json;
use std::{
    collections::HashMap,
    error::Error,
    fmt::Debug,
    future::Future,
    pin::Pin,
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
    Empty, FetchChunk, Query, ReferenceList, ReferenceRequest, ReferenceResponse, SendChunk,
};

mod serialization;
use serialization::*;

mod composite_plan;
use composite_plan::*;

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
    query_details: String,
}

impl DataFrameArtifact {
    pub fn new(df: DataFrame, policy: Policy) -> Self {
        DataFrameArtifact {
            dataframe: df,
            policy,
            fetchable: PolicyAction::Reject(String::from(
                "DataFrames uploaded by the Data Owner are protected.",
            )),
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

    fn get_df(&self, identifier: &str) -> Result<DelayedDataFrame, Status> {
        let dfs = self.dataframes.read().unwrap();
        let artifact = dfs.get(identifier).ok_or(Status::not_found(format!(
            "Could not find dataframe: identifier={}",
            identifier
        )))?;
        Ok(match &artifact.fetchable {
            PolicyAction::Accept => {
                let df = artifact.dataframe.clone();
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
                        Ok(dfs
                            .read()
                            .unwrap()
                            .get(&identifier)
                            .ok_or(Status::not_found(format!(
                                "Could not find dataframe: identifier={}",
                                identifier
                            )))?
                            .dataframe
                            .clone())
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

    fn get_policy_unchecked(&self, identifier: &str) -> Result<Policy, Status> {
        let dfs = self.dataframes.read().unwrap();
        Ok(dfs
            .get(identifier)
            .ok_or(Status::not_found(format!(
                "Could not find dataframe: identifier={}",
                identifier
            )))?
            .policy
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
    type FetchDataFrameStream = ReceiverStream<Result<FetchChunk, Status>>;

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
        let res = composite_plan.run(self)?;

        let header = get_df_header(&res.dataframe)?;
        let identifier = self.insert_df(res);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn send_data_frame(
        &self,
        request: Request<Streaming<SendChunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let df = df_artifact_from_stream(request.into_inner()).await?;

        let header = get_df_header(&df.dataframe)?;
        let identifier = self.insert_df(df);
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }

    async fn fetch_data_frame(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDataFrameStream>, Status> {
        let fut = {
            let df = self.get_df(&request.get_ref().identifier)?;
            stream_data(df, 32)
        };
        Ok(fut.await)

        // Ok(stream_data(df, 32).await)
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
        let identifier = String::from(&request.get_ref().identifier);
        let header = self.get_header(&identifier)?;

        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let state = BastionLabState::new();
    let addr = "0.0.0.0:50056".parse()?;
    println!("BastionLab server running...");
    Server::builder()
        .add_service(BastionLabServer::new(state))
        .serve(addr)
        .await?;
    Ok(())
}
