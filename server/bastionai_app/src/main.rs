use remote_torch::*;
use tonic::{transport::Server, Request, Response, Status};
use crate::reference_protocol_server::{ReferenceProtocol, ReferenceProtocolServer};
use uuid::Uuid;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use server::file_system::*;
use tch::*;
use http::Uri;
use std::net::SocketAddr;
use std::net::ToSocketAddrs;
use anyhow::{Context, Result};
use std::sync::Arc;
use std::collections::HashMap;


pub mod remote_torch {
    tonic::include_proto!("remote_torch");
}


struct Permission {
    owner: bool,
    user: bool,
}

struct Meta {
    size: usize,
    chunk_size: usize,
    nb_chunks: usize
}

pub struct Artifact<T> {
    permission: Permission,
    data: Vec<Vec<T>>,
    meta: Meta,
}

pub type ModuleType= Arc<HashMap<String,Artifact<TrainableCModule>>>;
pub type TensorType = Arc<HashMap<String,Artifact<Tensor>>>;

unsafe impl<T> Sync for Artifact<T>{}

#[derive(Default)]
pub(crate) struct MyReferenceProtocol {
    modules: ModuleType,
    tensors: TensorType,
}


impl MyReferenceProtocol {
    pub fn new(modules: ModuleType, tensors: TensorType) -> Self {
        Self{
            modules,
            tensors
        }
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

pub fn store_artifacts<'a, T>(
    artifacts: Vec<T>,
    path: &'a str,
    chunk_size: usize,
    single_mode: bool,
    mut data_store: &mut HashMap<String,Artifact<T>>
) -> Result<(), &'a str>
where
    T: Clone + Copy,
{
    let mut batch: Vec<T> = Vec::new();
    let mut index: usize = 0;
    let mut data: Vec<Vec<T>> = Vec::new();
    let permission = Permission {owner: true, user: true };

    if single_mode {
        let artifacts_list: Vec<T> = artifacts.clone().into_iter().map(|x| x).collect();

        if artifacts_list.len() != 1 {
            return Err("Error");
        }

        let meta = Meta { size: artifacts.len(), chunk_size, nb_chunks: 1 };
        data_store.insert(path.to_string(), Artifact {meta, permission, data: vec![artifacts]});
    } else {    
        for &artifact in &artifacts {
            if batch.len() < chunk_size {
                batch.push(artifact);
            } else {
                index += 1;
                data.push(batch);
                batch = vec![artifact];
            }
        }
        let meta = Meta { size: artifacts.len(), chunk_size, nb_chunks: 1 };
        
        data_store.insert(path.to_string(), Artifact{meta, permission, data});
        /* Spot for torch.save() */
    }
    Ok(())
}

#[tonic::async_trait]
impl ReferenceProtocol for MyReferenceProtocol {
    async fn send_data(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut tensors: Vec<Tensor> = Vec::new();
        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;

            data_bytes.append(&mut data_proto.data);
        }

        println!("Tensors Length: {}", data_bytes.len());
        
        while data_bytes.len() > 0 {
            let len:usize = as_u32_le(&data_bytes.drain(..4).collect::<Vec<u8>>()[..]) as usize;
            tensors.push(bytes_to_tensor(&data_bytes.drain(..len).collect::<Vec<u8>>()).unwrap());
        }

        println!("Tensors: {}", tensors.len());
        Ok(Response::new(Reference{
            identifier: String::from("id")
        }))
    }

    async fn send_model(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut module: Vec<TrainableCModule> = Vec::new();
        let identifier: Uuid = Uuid::new_v4();
        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;

            data_bytes.append(&mut data_proto.data);
        }

        println!("Module Length: {}", data_bytes.len());
        
        while data_bytes.len() > 0 {
            let len:usize = as_u32_le(&data_bytes.drain(..4).collect::<Vec<u8>>()[..]) as usize;
            println!("Length --> {}", len);
            module.push(bytes_to_module(&data_bytes.drain(..len).collect::<Vec<u8>>()).unwrap());
        }

        println!("Module: {:?}", &module);

        let chunk_size: usize = 64*1024;

        store_artifacts::<TrainableCModule>(module,&identifier.to_string(), chunk_size, true, &mut self.modules);
        Ok(Response::new(Reference{
            identifier: String::from("id")
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

    let protocol = MyReferenceProtocol::new(Arc::new(HashMap::new()), Arc::new(HashMap::new()));

    println!("BastionAI listening on {:?}", addr);
    Server::builder()
    .add_service(ReferenceProtocolServer::new(protocol))
    .serve(uri_to_socket(&addr)?)
    .await?;


    Ok(())

}
