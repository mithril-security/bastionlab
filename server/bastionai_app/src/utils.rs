use super::Chunk;
use crate::storage::{Artifact, SizedObjectsBytes};
use crate::Reference;
use tch::{TchError, Tensor};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::Request;
use tonic::{Response, Status};
use uuid::Uuid;

pub fn read_le_usize(input: &mut &[u8]) -> usize {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<usize>());
    *input = rest;
    usize::from_le_bytes(int_bytes.try_into().unwrap())
}

pub fn tcherror_to_status<T>(input: Result<T, TchError>) -> Result<T, Status> {
    input.map_err(|err| Status::internal(format!("Torch error: {}", err)))
}

pub async fn unstream_data(
    mut stream: tonic::Streaming<Chunk>,
) -> Result<Artifact<SizedObjectsBytes>, Status> {
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

pub async fn stream_data(
    artifact: Artifact<SizedObjectsBytes>,
    chunk_size: usize,
) -> Response<ReceiverStream<Result<Chunk, Status>>> {
    let (tx, rx) = mpsc::channel(4);

    let raw_bytes: Vec<u8> = artifact.data.into();
    tokio::spawn(async move {
        for (i, bytes) in raw_bytes.chunks(chunk_size).enumerate() {
            tx.send(Ok(Chunk {
                // Chunks always contain one object -> fix this
                data: bytes.to_vec(),
                description: if i == 0 {
                    artifact.description.clone()
                } else {
                    String::from("")
                },
                secret: vec![],
            }))
            .await
            .unwrap(); // Fix this
        }
    });

    Response::new(ReceiverStream::new(rx))
}

pub fn serialize_tensor(tensor: &Tensor) -> Vec<u8> {
    let capacity = tensor.numel() * tensor.f_kind().unwrap().elt_size_in_bytes();
    let mut bytes = vec![0; capacity];
    tensor.copy_data_u8(&mut bytes, tensor.numel());
    bytes
}

pub fn parse_reference(reference: Reference) -> Result<Uuid, Status> {
    Uuid::parse_str(&reference.identifier)
        .map_err(|_| Status::internal("Invalid BastionAI reference"))
}
