use super::{Chunk, ClientInfo};
use crate::storage::Artifact;
use bastionai_learning::serialization::SizedObjectsBytes;
use log::info;
use std::time::Instant;
use tch::Device;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Response, Status};
use ring::hmac;
use std::sync::{Arc, RwLock};

/// Returns a raw artifact from a stream of chunks received over gRPC.
/// 
/// This function only parses header data such as the name and description
/// of the artifact. The actual objects remains in binary format.
pub async fn unstream_data(
    mut stream: tonic::Streaming<Chunk>,
) -> Result<
    Artifact<SizedObjectsBytes>,
    Status,
> {
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut name: String = String::new();
    let mut description: String = String::new();
    let mut secret: Vec<u8> = Vec::new();
    let mut meta: Vec<u8> = Vec::new();
    let mut client_info: Option<ClientInfo> = None;

    let mut first = true;
    while let Some(chunk) = stream.next().await {
        let mut chunk = chunk?;
        data_bytes.append(&mut chunk.data);
        if first {
            first = false;
            name = chunk.name;
            description = chunk.description;
            secret = chunk.secret;
            client_info = chunk.client_info;
            meta = chunk.meta;
        }
    }

    Ok(Artifact {
        data: Arc::new(RwLock::new(data_bytes.into())),
        name,
        description,
        secret: hmac::Key::new(hmac::HMAC_SHA256, &secret),
        meta,
        client_info,
    })
}

/// Converts a raw artifact (a header and a binary object) into a stream of chunks to be sent over gRPC.
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
                meta: if i == 0 {
                    artifact.meta.clone()
                } else {
                    Vec::new()
                },
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

/// Parses a device string and returns a [`tch::Device`] object if the string is a valid device name.
pub fn parse_device(device: &str) -> Result<Device, Status> {
    Ok(match device {
        "cpu" => Device::Cpu,
        "gpu" => Device::cuda_if_available(),
        device => {
            if device.starts_with("cuda:") {
                let id = usize::from_str_radix(&device[5..], 10)
                    .or(Err(Status::invalid_argument("Unknown device")))?;
                Device::Cuda(id)
            } else {
                return Err(Status::invalid_argument("Unknown device"));
            }
        }
    })
}
