use http::Uri;
use remote_torch::*;
use tonic::{transport::Server, Request, Response, Status};
use crate::reference_protocol_server::{ReferenceProtocol, ReferenceProtocolServer};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tch::*;
use std::sync::{Arc};
use data_store::DataStore;
use uuid::Uuid;
use tokio::sync::mpsc;
use crate::utils::*;

mod utils;

pub mod remote_torch {
    tonic::include_proto!("remote_torch");
}

pub(crate) struct MyReferenceProtocol {
    data_store: Arc<DataStore>
}


impl MyReferenceProtocol {
    pub fn new(data_store: Arc<DataStore>) -> Self {
        Self{data_store}
    }
}


#[tonic::async_trait]
impl ReferenceProtocol for MyReferenceProtocol {
    async fn send_data(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut description: String = String::default();

        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;
            data_bytes.append(&mut data_proto.data);
            if description == String::default() {
                description = data_proto.description;
            }
        }

        let (tensors, tensors_bytes) = transform_bytes(data_bytes, bytes_to_tensor);

        println!("Tensors: {}", tensors.len());
        println!("Tensors Bytes: {}", tensors_bytes.len());
        println!("Tensors: {:#?}", tensors);
        let flat_byte_list = tensors_bytes.into_iter().flatten().collect::<Vec<u8>>();
        match self.data_store.add_batch_artifact(tensors, &flat_byte_list[..], description.as_str()) {
            Some(v) => Ok(Response::new(Reference{
                identifier: v.to_string()
            })),
            None => {return Err(Status::internal("Batch already uploaded!".to_string()))}
        }
    }

    async fn send_model(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {

        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut description: String = "".to_string();

        println!("Uploading model...");
        
        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;
            data_bytes.append(&mut data_proto.data);
            if description == "".to_string() {
                description = data_proto.description;
            }
        }
        println!("Length: {}", data_bytes.len());
        let (mut modules, module_bytes) = transform_bytes(data_bytes, bytes_to_module);

        let mut d = modules.drain(..);
        let first = d.next().unwrap();

        let flat_byte_list = module_bytes.into_iter().flatten().collect::<Vec<u8>>();

        match self.data_store.add_module_artifact(first, &flat_byte_list[..], description.as_str()) {
            Some(v) => {
                println!("Model upload done!");
                Ok(Response::new(Reference{
                identifier: v.to_string()
            }))},
            None => {return Err(Status::internal("Model already uploaded!".to_string()))}
        }
    }

    type FetchModelStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchDataSetStream = ReceiverStream<Result<Chunk, Status>>;

    async fn fetch_model(&self, request: Request<Reference>) -> Result<Response<Self::FetchModelStream>, Status> {
        let identifier = request.into_inner().identifier;
        let (tx, rx) = mpsc::channel(32);
        let res = self.data_store.get_model_with_identifier(Uuid::parse_str(&identifier).unwrap(), |artifact| {
            let tensors = artifact.get_data().method_is::<IValue>("trainable_parameters", &[]).unwrap();

            match tensors {
                IValue::TensorList(v) => {
                    Some(v)
                }
                _ => None
            }
        });
        
        match res {
            Some(v) => {
                println!("Module: ");
                tokio::spawn(
                    async move {
                        for tensor in v.unwrap() {
                            let serialized =  serialize_tensor(tensor.as_ref());
                            tx.send(Ok(Chunk{data: serialized, description: "".to_string()})).await.unwrap()
                        }
                    }
                );
                
            },
            None => {return Err(Status::internal("Model not available!".to_string()))}
        }

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn fetch_data_set(&self, request: Request<Reference>) -> Result<Response<Self::FetchDataSetStream>, Status> {
        let identifier = request.into_inner().identifier;
        let (tx, rx) = mpsc::channel(32);
        let res = self.data_store.get_dataset_with_identifier(Uuid::parse_str(&identifier).unwrap(), |artifact| {
            let tensors = artifact.get_data().get_tensors();
            let mut tensors_bytes: Vec<Vec<u8>> = Vec::new();

            for tensor in tensors {
                tensors_bytes.push(serialize_tensor(tensor));
            }
            Some(tensors_bytes)
        });

        match res {
            Some(v) => {
                tokio::spawn(
                    async move {
                        for bytes in v.as_ref().unwrap() {
                            tx.send(Ok(Chunk{data: bytes.clone(), description: "".to_string()})).await.unwrap()
                        }
                    }
                );
            },
            None => {return Err(Status::internal("Model not available!".to_string()))}
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn delete_model(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = request.into_inner().identifier;

        match self.data_store.delete_model(Uuid::parse_str(&identifier).unwrap()) {
            Some(_) => Ok(Response::new(Empty {})),
            None => {return Err(Status::internal("Failed to delete model!".to_string()))}
        }   
    }

    async fn delete_batch(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = request.into_inner().identifier;

        match self.data_store.delete_batch(Uuid::parse_str(&identifier).unwrap()) {
            Some(_) => Ok(Response::new(Empty {})),
            None => {return Err(Status::internal("Failed to delete model!".to_string()))}
        }       
    }

    async fn train(&self, _request: Request<TrainConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference {identifier: String::from("1")}))
    }

    async fn predict(&self, _request: Request<PredictConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference {identifier: String::from("1")}))
    }

    async fn test(&self, _request:Request<PredictConfig>) -> Result<Response<Accuracy>, Status> {
        Ok(Response::new(Accuracy::default()))
    }

    async fn get_available_models(&self, _request:Request<Empty>) -> Result<Response<AvailableObjects>, Status> {
        let res = get_available_objects(self.data_store.get_available_models());
        Ok(Response::new(AvailableObjects{available_objects: res}))
    }


    async fn get_available_data_sets(&self, _request:Request<Empty>) -> Result<Response<AvailableObjects>, Status> {
        let res = get_available_objects(self.data_store.get_available_datasets());
        Ok(Response::new(AvailableObjects{available_objects:res }))
    }

}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>{
    let addr: Uri = "[::1]:50051".parse::<Uri>()?;

    let protocol = MyReferenceProtocol::new(Arc::new(DataStore::new()));

    println!("BastionAI listening on {:?}", addr);
    Server::builder()
    .add_service(ReferenceProtocolServer::new(protocol))
    .serve(uri_to_socket(&addr)?)
    .await?;


    Ok(())

}
