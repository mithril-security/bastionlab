use tonic::{Streaming, Status, Response};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use super::Chunk;
use tch::{TchError, Device, IValue, Tensor, TrainableCModule, CModule};
use tch::nn::VarStore;
use crate::storage::{Artifact, SizedObjectsBytes};
use std::sync::Mutex;
use tokio::sync::mpsc;
use std::borrow::Borrow;
use std::convert::TryFrom;
use std::collections::VecDeque;
use uuid::Uuid;

pub fn read_le_usize(input: &mut &[u8]) -> usize {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<usize>());
    *input = rest;
    usize::from_le_bytes(int_bytes.try_into().unwrap())
}

pub fn tcherror_to_status<T>(input: Result<T, TchError>) -> Result<T, Status> {
    input.map_err(|err| Status::internal(format!("Torch error: {}", err)))
}

pub async fn unstream_data(mut stream: tonic::Streaming<Chunk>) -> Result<Artifact<SizedObjectsBytes>, Status> {
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut description: String = String::new();
    let mut secret: Vec<u8> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let mut chunk = chunk?;
        data_bytes.append(&mut chunk.data);
        if chunk.description.len() != 0 {
            description = chunk.description;
        }
        if chunk.secret.len() != 0 {
            secret = chunk.secret;
        }
    }

    Ok(Artifact::new(data_bytes.into(), description, &secret))
}

pub async fn stream_data(artifact: Artifact<SizedObjectsBytes>) -> Response<ReceiverStream<Result<Chunk, Status>>> {
    let (tx, rx) = mpsc::channel(4);
    
    for (i, bytes) in artifact.data.enumerate() {
        tx.send(Ok(Chunk { // Chunks always contain one object -> fix this
            data: bytes,
            description: if i == 0 { artifact.description.clone() } else { String::from("") },
            secret: vec![],
        })).await.unwrap() // Fix this
    }

    Response::new(ReceiverStream::new(rx))
}

// pub struct Module {
//     c_module: TrainableCModule,
//     var_store: VarStore,
// }

// impl Module {
//     fn new(c_module: TrainableCModule, var_store: VarStore) -> Self {
//         Module {
//             c_module,
//             var_store,
//         }
//     }
// }

// fn deserialize_module(mut data: &[u8]) -> Result<Module, TchError> {
//     let vs = VarStore::new(Device::Cpu);
//     Ok(Module::new(TrainableCModule::load_data(&mut data, vs.root())?, vs))
// }

// fn deserialize_batch_module(mut data: &[u8]) -> Result<Vec<Mutex<Tensor>>, TchError> {
//     let module = CModule::load_data(&mut data)?;

//     match module.method_is::<IValue>("data", &[])? {
//         IValue::TensorList(v) => Ok(v.into_iter().map(|x| Mutex::new(x)).collect()),
//         _ => Err(TchError::FileFormat(String::from("Invalid data, expected a batch module with a `data` function returning the actual data."))),
//     }
// }

pub fn serialize_tensor(tensor: &Tensor) -> Vec<u8> {
    let capacity = tensor.numel() * tensor.f_kind().unwrap().elt_size_in_bytes();
    let mut bytes = vec![0; capacity];
    tensor.copy_data_u8(&mut bytes, tensor.numel());
    bytes
}

// pub async fn dataset_artifact_from_stream(stream: tonic::Streaming<Chunk>) -> Result<Artifact, Status> {
//     let (nested_tensors, description, secret) = unstream_data(stream, deserialize_batch_module).await?;
//     let mut tensors = Vec::new();
//     for mut batch in nested_tensors {
//         tensors.append(&mut batch)
//     }
//     Ok(Artifact::new(ArtifactData::Dataset(tensors), description, &secret))
// }

// pub async fn module_artifact_from_stream(stream: tonic::Streaming<Chunk>) -> Result<Artifact, Status> {
//     let (mut module, description, secret) = unstream_data(stream, deserialize_module).await?;
//     let module = module.remove(0);
//     Ok(Artifact::new(ArtifactData::Module(module), description, &secret))
// }

// pub async fn module_to_stream(module: Module, description: &str) -> Result<Response<ReceiverStream<Result<Chunk, Status>>>, Status> {
//     let weights: Vec<Mutex<Tensor>> = tcherror_to_status(match tcherror_to_status(module.c_module.method_is::<IValue>("trainable_parameters", &[]))? {
//         IValue::TensorList(v) => Ok(v.into_iter().map(|x| Mutex::new(x)).collect()),
//         _ => Err(TchError::FileFormat(String::from("Invalid data, expected a batch module with a `data` function returning the actual data.")))
//     })?;
//     Ok(stream_data(&weights, description, serialize_tensor).await)
// }

// pub async fn dataset_to_stream(dataset: Vec<Mutex<Tensor>>, description: &str) -> Result<Response<ReceiverStream<Result<Chunk, Status>>>, Status> {
//     Ok(stream_data(&dataset, description, serialize_tensor).await)
// }
