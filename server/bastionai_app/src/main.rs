use tch::{Tensor, Device};
use tch::vision::dataset;
use tonic::{transport::Server, Request, Response, Status, Streaming};
use tokio_stream::wrappers::ReceiverStream;
use std::collections::HashMap;
use uuid::Uuid;
use std::sync::RwLock;
use private_learning::{SGD, l2_loss, Optimizer};

mod remote_torch {
    tonic::include_proto!("remote_torch");
}
use remote_torch::{Reference, References, Empty, Chunk, TrainConfig, TestConfig, Accuracy, Devices};
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
        let module: Artifact<Module> = tcherror_to_status(unstream_data(request.into_inner()).await?.deserialize())?;
        let description = String::from(module.description.clone());
        let identifier = Uuid::new_v4();
        
        self.modules.write().unwrap().insert(identifier.clone(), module);
        Ok(Response::new(Reference { identifier: format!("{}", identifier), description }))
    }

    async fn fetch_dataset(&self, request: Request<Reference>) -> Result<Response<Self::FetchDatasetStream>, Status> {
        let identifier = parse_reference(request.into_inner())?;
        let serialized = {
            let datasets = self.datasets.read().unwrap();
            let artifact = datasets.get(&identifier).ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 100_000_000).await)
    }

    async fn fetch_module(&self, request: Request<Reference>) -> Result<Response<Self::FetchModuleStream>, Status> {
        let identifier = parse_reference(request.into_inner())?; 
        let serialized = {
            let modules = self.modules.read().unwrap();
            let artifact = modules.get(&identifier).ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 100_000_000).await)
    }

    async fn delete_dataset(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = parse_reference(request.into_inner())?; 
        self.datasets.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn delete_module(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = parse_reference(request.into_inner())?; 
        self.modules.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn train(&self, request: Request<TrainConfig>) -> Result<Response<Empty>, Status> {
        let config = request.into_inner();
        let dataset_id = parse_reference(config.dataset.clone().ok_or(Status::invalid_argument("Not found"))?)?;
        let module_id = parse_reference(config.model.clone().ok_or(Status::invalid_argument("Not found"))?)?;
        let device = parse_device(&config.device)?;
        {
            let datasets = self.datasets.read().unwrap();
            let mut modules = self.modules.write().unwrap();
            let dataset = datasets.get(&dataset_id).ok_or(Status::not_found("Not found"))?;
            let module = modules.get_mut(&module_id).ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(module.data.write().unwrap().train(&dataset.data.read().unwrap(), config, device))?;
        }
        Ok(Response::new(Empty {}))
    }

    async fn test(&self, request: Request<TestConfig>) -> Result<Response<Accuracy>, Status> {
        let config = request.into_inner();
        let dataset_id = parse_reference(config.dataset.clone().ok_or(Status::invalid_argument("Not found"))?)?;
        let module_id = parse_reference(config.model.clone().ok_or(Status::invalid_argument("Not found"))?)?;
        let device = parse_device(&config.device)?;
        let accuracy = {
            let datasets = self.datasets.read().unwrap();
            let modules = self.modules.read().unwrap();
            let dataset = datasets.get(&dataset_id).ok_or(Status::not_found("Not found"))?;
            let module = modules.get(&module_id).ok_or(Status::not_found("Not found"))?;
            let acc = tcherror_to_status(module.data.write().unwrap().test(&dataset.data.read().unwrap(), config, device))?;
            acc
        };
        Ok(Response::new(Accuracy { value: accuracy }))
    }

    async fn available_models(&self, _request: Request<Empty>) -> Result<Response<References>, Status> {
        let list = self.modules.read().unwrap().iter().map(|(k, v)| Reference { identifier: format!("{}", k), description: v.description.clone() }).collect();
        
        Ok(Response::new(References { list } ))
    }

    async fn available_datasets(&self, _request: Request<Empty>) -> Result<Response<References>, Status> {
        let list = self.datasets.read().unwrap().iter().map(|(k, v)| Reference { identifier: format!("{}", k), description: v.description.clone() }).collect();
        
        Ok(Response::new(References { list } ))
    }

    async fn available_devices(&self, _request: Request<Empty>) -> Result<Response<Devices>, Status> {
        let mut list = vec![String::from("cpu")];
        if tch::Cuda::is_available() {
            list.push(String::from("gpu"));
            for index in 0..tch::Cuda::device_count()  {
                list.push(format!("cuda:{}", index));
            }
        }
        
        Ok(Response::new(Devices { list } ))
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
