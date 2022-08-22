use super::ClientInfo;
use crate::remote_torch::{train_config, Metric, TestConfig, TrainConfig};
use crate::telemetry::{self, TelemetryEventProps};
use crate::utils::tcherror_to_status;
use bastionai_learning::data::Dataset;
use bastionai_learning::nn::{LossType, Module};
use bastionai_learning::optim::{Adam, Optimizer, SGD};
use bastionai_learning::procedures;
use log::info;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tch::{Device, TchError};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Status};

fn build_train_context(
    module: Arc<RwLock<Module>>,
    config: TrainConfig,
) -> Result<(Box<dyn Optimizer + Send>, procedures::Metric), TchError> {
    let parameters = match config
        .privacy
        .ok_or(TchError::FileFormat(String::from("Invalid privacy option")))?
    {
        train_config::Privacy::Standard(_) => module.read().unwrap().parameters(),
        train_config::Privacy::DifferentialPrivacy(train_config::DpParameters {
            max_grad_norm,
            noise_multiplier,
        }) => module.read().unwrap().private_parameters(
            max_grad_norm as f64,
            noise_multiplier as f64,
            LossType::Mean(config.batch_size as i64),
        ),
    };

    let optimizer = match config
        .optimizer
        .ok_or(TchError::FileFormat(String::from("Invalid optimizer")))?
    {
        train_config::Optimizer::Sgd(train_config::Sgd {
            learning_rate,
            weight_decay,
            momentum,
            dampening,
            nesterov,
        }) => Box::new(
            SGD::new(parameters, learning_rate as f64)
                .weight_decay(weight_decay as f64)
                .momentum(momentum as f64)
                .dampening(dampening as f64)
                .nesterov(nesterov),
        ) as Box<dyn Optimizer + Send>,
        train_config::Optimizer::Adam(train_config::Adam {
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            amsgrad,
        }) => Box::new(
            Adam::new(parameters, learning_rate as f64)
                .beta_1(beta_1 as f64)
                .beta_2(beta_2 as f64)
                .epsilon(epsilon as f64)
                .weight_decay(weight_decay as f64)
                .amsgrad(amsgrad),
        ) as Box<dyn Optimizer + Send>,
    };

    let metric = procedures::Metric::try_from_name(&config.metric)?;

    Ok((optimizer, metric))
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
        let epochs = config.epochs;
        let batch_size = config.batch_size;
        let start_time = Instant::now();
        let (model_hash, client_info) = &*model_client_info;
        let (dataset_hash, _) = &*dataset_client_info;

        match tcherror_to_status(build_train_context(Arc::clone(&module), config)) {
            Ok((optimizer, metric)) => {
                let trainer = Module::train(
                    module,
                    dataset,
                    optimizer,
                    metric,
                    epochs as usize,
                    batch_size as usize,
                    device,
                );
                let nb_epochs = trainer.nb_epochs() as i32;
                let nb_batches = trainer.nb_batches() as i32;

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
        let metric = tcherror_to_status(procedures::Metric::try_from_name(&config.metric));
        match metric {
            Ok(metric) => {
                let tester =
                    Module::test(module, dataset, metric, config.batch_size as usize, device);
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
