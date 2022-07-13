use remote_torch::*;
use tonic::{transport::Server, Request, Response, Status};
use crate::reference_protocol_server::{ReferenceProtocol, ReferenceProtocolServer};
use uuid::Uuid;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tch::*;
use http::Uri;
use std::net::SocketAddr;
use std::net::ToSocketAddrs;
use anyhow::{Context, Result};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::RwLock;
use data_store::DataStore;


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

#[tonic::async_trait]
impl ReferenceProtocol for MyReferenceProtocol {
    async fn send_data(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut tensors_bytes: Vec<Vec<u8>> = Vec::new();

        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;

            data_bytes.append(&mut data_proto.data);
        }

        println!("Tensors Length: {}", data_bytes.len());
        
        while data_bytes.len() > 0 {
            let len:usize = as_u32_le(&data_bytes.drain(..4).collect::<Vec<u8>>()[..]) as usize;
            let bytes = &data_bytes.drain(..len).collect::<Vec<u8>>();
            tensors_bytes.push(bytes.clone());
            tensors.push(bytes_to_tensor(&bytes[..]).unwrap());
        }

        println!("Tensors: {}", tensors.len());

        let flat_byte_list = tensors_bytes.into_iter().flatten().collect::<Vec<u8>>();
        let identifier = self.data_store.add_batch_artifact(Uuid::new_v4(), tensors, &flat_byte_list[..]).unwrap();

        Ok(Response::new(Reference{
            identifier
        }))
    }

    async fn send_model(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut module: Vec<TrainableCModule> = Vec::new();
        let mut module_bytes: Vec<Vec<u8>> = Vec::new();

         while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;

            data_bytes.append(&mut data_proto.data);
        }
        
        while data_bytes.len() > 0 {
            let len:usize = as_u32_le(&data_bytes.drain(..4).collect::<Vec<u8>>()[..]) as usize;
            println!("Length --> {}", len);
            let bytes= &data_bytes.drain(..len).collect::<Vec<u8>>();
            module_bytes.push(bytes.clone());
            module.push(bytes_to_module(bytes).unwrap());
            
        }

        let mut d = module.drain(..);
        let first = d.next().unwrap();

        let flat_byte_list = module_bytes.into_iter().flatten().collect::<Vec<u8>>();

        let identifier = self.data_store.add_module_artifact(Uuid::new_v4(), first, &flat_byte_list[..]).unwrap();

        Ok(Response::new(Reference{
            identifier: identifier
        }))
    }

    type FetchStream = ReceiverStream<Result<Chunk, Status>>;

    async fn fetch(&self, request: Request<Reference>) -> Result<Response<Self::FetchStream>, Status> {
        unimplemented!()
    }

    async fn delete(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {

        Ok(Response::new(Empty{}))
    }

    async fn mode(&self, request: Request<ReferenceMode>) -> Result<Response<Empty>, Status> {
        Ok(Response::new(Empty{}))
    }

    async fn group_add(&self, request: Request<Users>) -> Result<Response<Empty>, Status> {
        Ok(Response::new(Empty{}))
    }

    async fn group_del(&self, request: Request<Users>) -> Result<Response<Empty>, Status> {
        Ok(Response::new(Empty{}))
    }

    async fn train(&self, request: Request<TrainConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference {identifier: String::from("1")}))
    }

    async fn predict(&self, request: Request<PredictConfig>) -> Result<Response<Reference>, Status> {
        Ok(Response::new(Reference {identifier: String::from("1")}))
    }

    async fn test(&self, request:Request<PredictConfig>) -> Result<Response<Accuracy>, Status> {
        Ok(Response::new(Accuracy::default()))
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
