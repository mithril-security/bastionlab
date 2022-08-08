use super::Chunk;
use crate::storage::{Artifact, SizedObjectsBytes};
use crate::Reference;
use tch::{TchError, Device};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
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

    let raw_bytes: Vec<u8> = artifact.data.into_inner().unwrap().into();
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

pub fn parse_reference(reference: Reference) -> Result<Uuid, Status> {
    Uuid::parse_str(&reference.identifier)
        .map_err(|_| Status::internal("Invalid BastionAI reference"))
}

pub fn parse_device(device: &str) -> Result<Device, Status> {
    Ok(match device {
        "cpu" => Device::Cpu,
        "gpu" => Device::cuda_if_available(),
        device => {
            if device.starts_with("cuda:") {
                let id = usize::from_str_radix(&device[5..], 10).or(Err(Status::invalid_argument("Wrong device")))?;
                Device::Cuda(id)
            } else {
                return Err(Status::invalid_argument("Wrong device"));
            }
        }
    })
}
