use super::{Chunk, ClientInfo};
use crate::storage::Artifact;
use crate::Reference;
use bastionai_learning::serialization::SizedObjectsBytes;
use log::info;
use std::{sync::Arc, time::Instant};
use tch::Device;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Response, Status};
use uuid::Uuid;

pub async fn unstream_data(
    mut stream: tonic::Streaming<Chunk>,
) -> Result<
    (
        Artifact<SizedObjectsBytes>,
        Option<ClientInfo>
    ),
    Status,
> {
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut name: String = String::new();
    let mut description: String = String::new();
    let mut secret: Vec<u8> = Vec::new();
    let mut client_info: Option<ClientInfo> = None;

    while let Some(chunk) = stream.next().await {
        let mut chunk = chunk?;
        data_bytes.append(&mut chunk.data);
        if chunk.name.len() != 0 {
            name = chunk.name;
        }
        if chunk.description.len() != 0 {
            description = chunk.description;
            client_info = chunk.client_info;
        }
        if chunk.secret.len() != 0 {
            secret = chunk.secret;
        }
    }

    Ok((
        Artifact::new(data_bytes.into(), name, description, &secret),
        client_info,
    ))
}

pub async fn stream_data(
    artifact: Artifact<SizedObjectsBytes>,
    chunk_size: usize,
    stream_type: String,
) -> Response<ReceiverStream<Result<Chunk, Status>>> {
    let (tx, rx) = mpsc::channel(4);

    let raw_bytes: Vec<u8> = Arc::try_unwrap(artifact.data)
        .unwrap()
        .into_inner()
        .unwrap()
        .into();
    let start_time = Instant::now();
    tokio::spawn(async move {
        for (i, bytes) in raw_bytes.chunks(chunk_size).enumerate() {
            tx.send(Ok(Chunk {
                // Chunks always contain one object -> fix this
                data: bytes.to_vec(),
                name: if i == 0 {
                    artifact.name.clone()
                } else {
                    String::from("")
                },
                description: if i == 0 {
                    artifact.description.clone()
                } else {
                    String::from("")
                },
                secret: vec![],
                client_info: Some(ClientInfo::default()),
            }))
            .await
            .unwrap(); // Fix this
        }
    });

    info!(
    target: "BastionAI",
            "{}",
                format!("{} fetched successfully in {}ms", stream_type,
                start_time.elapsed().as_millis())
            );

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
                let id = usize::from_str_radix(&device[5..], 10)
                    .or(Err(Status::invalid_argument("Wrong device")))?;
                Device::Cuda(id)
            } else {
                return Err(Status::invalid_argument("Wrong device"));
            }
        }
    })
}
