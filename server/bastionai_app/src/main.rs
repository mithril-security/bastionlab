use tonic::{transport::Server, Request, Response, Status, Streaming};
use tokio_stream::wrappers::ReceiverStream;
use std::collections::HashMap;
use uuid::Uuid;
use std::sync::RwLock;

mod remote_torch {
    tonic::include_proto!("remote_torch");
}
use remote_torch::{Reference, References, Empty, Chunk, TrainConfig, PredictConfig, Accuracy};
use remote_torch::remote_torch_server::{RemoteTorch, RemoteTorchServer};

mod storage;
use storage::{Artifact, Module, Dataset};

mod utils;
use utils::*;

struct BastionAIServer {
    modules: RwLock<HashMap<Uuid, Artifact<Module>>>,
    datasets: RwLock<HashMap<Uuid, Artifact<Dataset>>>,
}

impl BastionAIServer {
    pub fn new() -> Self {
        BastionAIServer {
            modules: RwLock::new(HashMap::new()),
            datasets: RwLock::new(HashMap::new()),
        }
    }
}

#[tonic::async_trait]
impl RemoteTorch for BastionAIServer {
    type FetchDatasetStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchModuleStream = ReceiverStream<Result<Chunk, Status>>;

    async fn send_dataset(&self, request: Request<Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let dataset: Artifact<Dataset> = tcherror_to_status((unstream_data(request.into_inner()).await?).deserialize())?;
        let description = String::from(dataset.description.clone());
        let identifier = Uuid::new_v4();
        
        self.datasets.write().unwrap().insert(identifier.clone(), dataset);

        Ok(Response::new(Reference { identifier: format!("{}", identifier), description }))
    }

    async fn send_model(&self, request: Request<Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let module: Artifact<Module> = tcherror_to_status((unstream_data(request.into_inner()).await?).deserialize())?;
        let description = String::from(module.description.clone());
        let identifier = Uuid::new_v4();
        
        self.modules.write().unwrap().insert(identifier.clone(), module);

        Ok(Response::new(Reference { identifier: format!("{}", identifier), description }))
    }

    async fn fetch_dataset(&self, request: Request<Reference>) -> Result<Response<Self::FetchDatasetStream>, Status> {
        let identifier = Uuid::parse_str(&request.into_inner().identifier).unwrap(); // Fix this;
        let serialized = {
            let datasets = self.datasets.read().unwrap();
            let artifact = datasets.get(&identifier).ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized).await)
    }

    async fn fetch_module(&self, request: Request<Reference>) -> Result<Response<Self::FetchModuleStream>, Status> {
        let identifier = Uuid::parse_str(&request.into_inner().identifier).unwrap(); // Fix this;
        let serialized = {
            let modules = self.modules.read().unwrap();
            let artifact = modules.get(&identifier).ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized).await)
    }

    async fn delete_dataset(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = Uuid::parse_str(&request.into_inner().identifier).unwrap(); // Fix this;
        self.datasets.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn delete_module(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = Uuid::parse_str(&request.into_inner().identifier).unwrap(); // Fix this;
        self.modules.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn train(&self, request: Request<TrainConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference { identifier: String::from(""), description: String::from("") }))
    }

    async fn predict(&self, request: Request<PredictConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference { identifier: String::from(""), description: String::from("") }))
    }

    async fn test(&self, request:Request<PredictConfig>) -> Result<Response<Accuracy>, Status> {
        Ok(Response::new(Accuracy { value: 0. }))
    }

    async fn available_models(&self, request: Request<Empty>) -> Result<Response<References>, Status> {
        let list = self.modules.read().unwrap().iter().map(|(k, v)| Reference { identifier: format!("{}", k), description: v.description.clone() }).collect();
        
        Ok(Response::new(References { list } ))
    }

    async fn available_datasets(&self, _request:Request<Empty>) -> Result<Response<References>, Status> {
        let list = self.datasets.read().unwrap().iter().map(|(k, v)| Reference { identifier: format!("{}", k), description: v.description.clone() }).collect();
        
        Ok(Response::new(References { list } ))
    }

}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let server = BastionAIServer::new();

    println!("BastionAI listening on {:?}", addr);
    Server::builder()
        .add_service(RemoteTorchServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
