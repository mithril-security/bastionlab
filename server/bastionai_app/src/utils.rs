use super::Chunk;
use crate::remote_torch::{ClientInfo, Metric, TestConfig, TrainConfig};
use crate::storage::{Artifact, Dataset, Module, SizedObjectsBytes};
use crate::telemetry::TelemetryEventProps;
use crate::{telemetry, Reference};
use log::info;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tch::{Device, TchError};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{client, Response, Status};
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
) -> Result<
    (
        Artifact<SizedObjectsBytes>,
        Option<ClientInfo>,
        String,
        String,
    ),
    Status,
> {
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut description: String = String::new();
    let mut secret: Vec<u8> = Vec::new();
    let mut client_info: Option<ClientInfo> = None;
    let mut dataset_name: String = String::new();
    let mut model_name: String = String::new();

    while let Some(chunk) = stream.next().await {
        let mut chunk = chunk?;
        data_bytes.append(&mut chunk.data);
        if chunk.description.len() != 0 {
            description = chunk.description;
            client_info = chunk.client_info;
            dataset_name = chunk.dataset_name;
            model_name = chunk.model_name;
        }
        if chunk.secret.len() != 0 {
            secret = chunk.secret;
        }
    }

    Ok((
        Artifact::new(data_bytes.into(), description, &secret),
        client_info,
        dataset_name,
        model_name,
    ))
}

pub async fn stream_module_train(
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    config: TrainConfig,
    device: Device,
    model_client_info: Arc<(String, Option<ClientInfo>)>,
    dataset_client_info: Arc<(String, Option<ClientInfo>)>,
) -> Response<ReceiverStream<Result<Metric, Status>>> {
    let (tx, rx) = mpsc::channel(1);
    tokio::spawn(async move {
        let trainer = tcherror_to_status(Module::train(module, dataset, config, device));
        match trainer {
            Ok(trainer) => {
                let nb_epochs = trainer.nb_epochs() as i32;
                let nb_batches = trainer.nb_batches() as i32;
                let (model_hash, client_info) = &*model_client_info;
                let (dataset_hash, _) = &*dataset_client_info;
                let start_time = Instant::now();
                telemetry::add_event(
                    TelemetryEventProps::TrainerLog {
                        log_type: Some("start_training".to_string()),
                        model_hash: Some(model_hash.clone()),
                        dataset_hash: Some(dataset_hash.clone()),
                        time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    },
                    client_info.clone(),
                );
                for res in trainer {
                    let res = tcherror_to_status(res.map(|(epoch, batch, value)| Metric {
                        epoch,
                        batch,
                        value,
                        nb_epochs,
                        nb_batches,
                    }));
                    tx.send(res).await.unwrap(); // Fix this
                }

                telemetry::add_event(
                    TelemetryEventProps::TrainerLog {
                        log_type: Some("end_training".to_string()),
                        model_hash: Some(model_hash.clone()),
                        dataset_hash: Some(dataset_hash.clone()),
                        time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    },
                    client_info.clone(),
                );
                info!(
                target: "BastionAI",
                            "Model trained successfully in {}ms",
                            start_time.elapsed().as_millis()
                        );
            }
            Err(e) => tx.send(Err(e)).await.unwrap(), // Fix this
        }
    });

    Response::new(ReceiverStream::new(rx))
}

pub async fn stream_module_test(
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    config: TestConfig,
    device: Device,
    model_client_info: Arc<(String, Option<ClientInfo>)>,
    dataset_client_info: Arc<(String, Option<ClientInfo>)>,
) -> Response<ReceiverStream<Result<Metric, Status>>> {
    let (tx, rx) = mpsc::channel(1);
    tokio::spawn(async move {
        let tester = tcherror_to_status(Module::test(module, dataset, config, device));
        match tester {
            Ok(tester) => {
                let nb_batches = tester.nb_batches() as i32;
                let (model_hash, client_info) = &*model_client_info;
                let (dataset_hash, _) = &*dataset_client_info;

                let start_time = Instant::now();
                telemetry::add_event(
                    TelemetryEventProps::TrainerLog {
                        log_type: Some("start_testing".to_string()),
                        model_hash: Some(model_hash.clone()),
                        dataset_hash: Some(dataset_hash.clone()),
                        time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    },
                    client_info.clone(),
                );
                for res in tester {
                    let res = tcherror_to_status(res.map(|(batch, value)| Metric {
                        epoch: 0,
                        batch,
                        value,
                        nb_epochs: 1,
                        nb_batches,
                    }));
                    tx.send(res).await.unwrap(); // Fix this
                }
                telemetry::add_event(
                    TelemetryEventProps::TrainerLog {
                        log_type: Some("end_testing".to_string()),
                        model_hash: Some(model_hash.clone()),
                        dataset_hash: Some(dataset_hash.clone()),
                        time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    },
                    client_info.clone(),
                );
                info!(
                target: "BastionAI",
                            "Model tested successfully in {}ms",
                            start_time.elapsed().as_millis()
                        );
            }
            Err(e) => tx.send(Err(e)).await.unwrap(), // Fix this
        }
    });

    Response::new(ReceiverStream::new(rx))
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
                description: if i == 0 {
                    artifact.description.clone()
                } else {
                    String::from("")
                },
                dataset_name: "".to_string(),
                model_name: "".to_string(),
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

pub fn fill_blank_and_print(content: &str, size: usize) {
    let trail_char = "#";
    let trail: String = trail_char.repeat((size - 2 - content.len()) / 2);
    let trail2: String =
        trail_char.repeat(((size - 2 - content.len()) as f32 / 2.0).ceil() as usize);
    println!("{} {} {}", trail, content, trail2);
}
