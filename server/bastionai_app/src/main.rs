use remote_torch::*;
use tonic::{transport::Server, Request, Response, Status};
use crate::reference_protocol_server::ReferenceProtocol;
use uuid::Uuid;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use server::file_system::*;
use tch::*;

pub mod remote_torch {
    tonic::include_proto!("remote_torch");
}

#[derive(Debug, Default)]
pub struct MyReferenceProtocol {}


/*
    Write a function that takes bytes and returns tch.Module
    Write a function that takes bytes and returns tch.Tensor
*/

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

// fn bytes_to_tensor(mut data: &[u8]) -> Result<Tensor, TchError> {
//     let tensors = Tensor::load_from_stream(data.to_vec().)?;
// }

#[tonic::async_trait]
impl ReferenceProtocol for MyReferenceProtocol {
    async fn send_data(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
        let identifier: String = format!("data/{}", Uuid::new_v4());
        
        let mut stream = request.into_inner();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut data_length: u32 = 0;
        while let Some(data_stream) = stream.next().await {
            let mut data_proto = data_stream?;
            if data_length == 0 {
                data_length = data_proto.length as u32;
            }
            data_bytes.append(&mut data_proto.data);
        }
        // store_artifacts(artifacts: Vec::new(), )
        let mut _item: usize = 0;

        while data_bytes.len() > 0 {
            let _value:usize = as_u32_le(&data_bytes[0..3]) as usize;
            data_bytes = data_bytes.clone().into_iter().skip(4).collect();

            bytes_to_module(&data_bytes.clone().into_iter().take(_value).collect::<Vec<u8>>()[..]);
            data_bytes = data_bytes.clone().into_iter().skip(_value).collect();

            _item += _value;
        }
        Ok(Response::new(Reference{
            identifier: String::from("id")
        }))
    }

    async fn send_model(&self, request: Request<tonic::Streaming<Chunk>>) -> Result<Response<Reference>, Status> {
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

fn main() {
    println!("Hello, world!");
}
