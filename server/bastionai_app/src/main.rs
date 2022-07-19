use remote_torch::*;
use tonic::{transport::Server, Request, Response, Status};
use crate::reference_protocol_server::{ReferenceProtocol, ReferenceProtocolServer};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tch::*;
use http::Uri;
use std::net::SocketAddr;
use std::net::ToSocketAddrs;
use anyhow::{Context, Result};
use std::sync::Arc;
use data_store::DataStore;
use uuid::Uuid;

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

fn as_u32_le(array: &[u8]) -> u32 {
    ((array[0] as u32) <<  0) +
    ((array[1] as u32) <<  8) +
    ((array[2] as u32) << 16) +
    ((array[3] as u32) << 24)
}

fn bytes_to_module(mut data: &[u8]) -> Result<tch::TrainableCModule, TchError> {
    let module:TrainableCModule  = TrainableCModule::load_data(&mut data, nn::VarStore::new(Device::Cpu).root())?;
    Ok(module)
}

fn bytes_to_tensor(data: &[u8]) -> Result<Tensor, TchError> {
    let tensor = Tensor::of_slice(data);
    Ok(tensor)
}

fn transform_bytes<T>(mut data: Vec<u8>, func: impl Fn(&[u8]) -> Result<T, TchError>) -> (Vec<T>, Vec<Vec<u8>>) {
    let mut transformed: Vec<T> = Vec::new();
    let mut ret_bytes: Vec<Vec<u8>> = Vec::new();

    while data.len() > 0 {
        let len:usize = as_u32_le(&data.drain(..4).collect::<Vec<u8>>()[..]) as usize;
        let bytes = &data.drain(..len).collect::<Vec<u8>>();
        ret_bytes.push(bytes.clone());
        transformed.push(func(&bytes[..]).unwrap());
    }

    (transformed, ret_bytes)
}

fn get_available_objects(objects: Vec<(String, String)>) -> Vec<AvailableObject> {
    let res:Vec<AvailableObject> = objects.into_iter()
    .map(|(k,v)|{AvailableObject{reference: k.to_string(), description: v.to_string()}})
    .collect::<Vec<AvailableObject>>();
    res
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

        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;
            data_bytes.append(&mut data_proto.data);
            if description == "".to_string() {
                description = data_proto.description;
            }
        }
        
        let (mut modules, module_bytes) = transform_bytes(data_bytes, bytes_to_module);

        let mut d = modules.drain(..);
        let first = d.next().unwrap();

        let flat_byte_list = module_bytes.into_iter().flatten().collect::<Vec<u8>>();

        match self.data_store.add_module_artifact(first, &flat_byte_list[..], description.as_str()) {
            Some(v) => Ok(Response::new(Reference{
                identifier: v.to_string()
            })),
            None => {return Err(Status::internal("Model already uploaded!".to_string()))}
        }
    }

    type FetchStream = ReceiverStream<Result<Chunk, Status>>;

    async fn fetch(&self, request: Request<Reference>) -> Result<Response<Self::FetchStream>, Status> {
        let identifier = request.into_inner().identifier;

        let res = self.data_store.get_model_with_identifier(Uuid::parse_str(&identifier).unwrap(),  |artifact| {
            println!("Artifact: {:?}", artifact.get_data());
        });
        
        match res {
            Some(v) => {
                println!("Module: {:?}", v);
            },
            None => {return Err(Status::internal("Model not uploaded!".to_string()))}
        }

        unimplemented!()
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
        }       }


    async fn train(&self, request: Request<TrainConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference {identifier: String::from("1")}))
    }

    async fn predict(&self, request: Request<PredictConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference {identifier: String::from("1")}))
    }

    async fn test(&self, request:Request<PredictConfig>) -> Result<Response<Accuracy>, Status> {
        Ok(Response::new(Accuracy::default()))
    }

    async fn get_available_models(&self, _request:Request<Empty>) -> Result<Response<AvailableObjects>, Status> {
        let res = get_available_objects(self.data_store.get_available_models());
        Ok(Response::new(AvailableObjects{available_models: res}))
    }


    async fn get_available_data_sets(&self, _request:Request<Empty>) -> Result<Response<AvailableObjects>, Status> {
        let res = get_available_objects(self.data_store.get_available_datasets());
        Ok(Response::new(AvailableObjects{available_models:res }))
    }

}

fn uri_to_socket(uri: &Uri) -> Result<SocketAddr> {
    uri.authority()
        .context("No authority")?
        .as_str()
        .to_socket_addrs()?
        .next()
        .context("Uri could not be converted to socket")
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>{
    let addr: Uri = "[::1]:50052".parse::<Uri>()?;

    let protocol = MyReferenceProtocol::new(Arc::new(DataStore::new()));

    println!("BastionAI listening on {:?}", addr);
    Server::builder()
    .add_service(ReferenceProtocolServer::new(protocol))
    .serve(uri_to_socket(&addr)?)
    .await?;


    Ok(())

}
