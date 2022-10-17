use super::ClientInfo;
use crate::remote_torch::{train_config, Metric, TestConfig, TrainConfig};
use crate::telemetry::{self, TelemetryEventProps};
use crate::utils::tcherror_to_status;
use crate::CheckPoint;
use bastionai_learning::data::privacy_guard::PrivacyBudget;
use bastionai_learning::data::Dataset;
use bastionai_learning::nn::{Forward, LossType, Module, Parameters};
use bastionai_learning::optim::{Adam, Optimizer, OptimizerStateType, SGD};
use bastionai_learning::procedures::{self, Tester, Trainer};
use bastionai_learning::serialization::BinaryModule;

use log::info;
use std::io::Cursor;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tch::{Device, TchError, Tensor};
use tonic::Status;

#[derive(Debug)]
pub enum Run {
    Ok(Metric),
    Error(Status),
    Pending,
}

/// Returns a metric by name from config and computes per step privacy budget for metrics
fn build_shared_context(
    metric: &str,
    metric_eps: f32,
    batch_size: i32,
    dataset_size: usize,
    nb_epochs: i32,
) -> Result<(procedures::Metric, PrivacyBudget), TchError> {
    let total_nb_batches = dataset_size as i32 / batch_size * nb_epochs;
    let metric = procedures::Metric::try_from_name(metric)?;
    let metric_budget = if metric_eps < 0.0 {
        PrivacyBudget::NotPrivate
    } else {
        PrivacyBudget::Private(metric_eps / total_nb_batches as f32)
    };

    Ok((metric, metric_budget))
}

/// Returns a forward pass, a metric and a metric budget from config.
fn build_test_context<'a>(
    module: &'a mut Module,
    dataset: &Dataset,
    config: TestConfig,
    private: bool,
) -> Result<
    (
        Forward<'a>,
        procedures::Metric,
        PrivacyBudget,
        Parameters<'a>,
    ),
    TchError,
> {
    let (forward, params) = if private {
        module.private_parameters(0.0, 0.0, LossType::Sum)
    } else {
        module.parameters()
    };
    let (metric, metric_budget) = build_shared_context(
        &config.metric,
        config.metric_eps,
        config.batch_size,
        dataset.len(),
        1,
    )?;

    Ok((forward, metric, metric_budget, params))
}

/// Returns a forward pass, an optimizer, a metric and a metric budget from config.
fn build_train_context<'a>(
    module: &'a mut Module,
    dataset: &Dataset,
    config: TrainConfig,
    optimizer_state: &Option<OptimizerStateType>,
    weights: &[u8],
) -> Result<
    (
        Forward<'a>,
        Box<dyn Optimizer + 'a>,
        procedures::Metric,
        PrivacyBudget,
    ),
    TchError,
> {
    let q = config.batch_size as f32 / dataset.len() as f32;
    let t = config.epochs as f32 / q;
    let (forward, parameters) = if config.eps < 0.0 {
        module.parameters()
    } else {
        module.private_parameters(
            config.eps / (q * t.sqrt()),
            config.max_grad_norm,
            LossType::Mean(config.batch_size as i64),
        )
    };

    let optimizer = match config
        .clone()
        .optimizer
        .ok_or(TchError::FileFormat(String::from("Invalid optimizer")))?
    {
        train_config::Optimizer::Sgd(train_config::Sgd {
            learning_rate,
            weight_decay,
            momentum,
            dampening,
            nesterov,
        }) => {
            if config.resume && optimizer_state.is_some() {
                Box::new(SGD::load_from_checkpoint(
                    optimizer_state,
                    weights,
                    learning_rate as f64,
                    parameters,
                )?) as Box<dyn Optimizer + 'a>
            } else {
                Box::new(
                    SGD::new(parameters, learning_rate as f64)
                        .weight_decay(weight_decay as f64)
                        .momentum(momentum as f64)
                        .dampening(dampening as f64)
                        .nesterov(nesterov),
                ) as Box<dyn Optimizer + 'a>
            }
        }
        train_config::Optimizer::Adam(train_config::Adam {
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            amsgrad,
        }) => {
            if config.resume {
                Box::new(
                    Adam::new(parameters, learning_rate as f64)
                        .beta_1(beta_1 as f64)
                        .beta_2(beta_2 as f64)
                        .epsilon(epsilon as f64)
                        .weight_decay(weight_decay as f64)
                        .amsgrad(amsgrad),
                ) as Box<dyn Optimizer + 'a>
            } else {
                Box::new(
                    Adam::new(parameters, learning_rate as f64)
                        .beta_1(beta_1 as f64)
                        .beta_2(beta_2 as f64)
                        .epsilon(epsilon as f64)
                        .weight_decay(weight_decay as f64)
                        .amsgrad(amsgrad),
                ) as Box<dyn Optimizer + 'a>
            }
        }
    };

    let (metric, metric_budget) = build_shared_context(
        &config.metric,
        config.metric_eps,
        config.batch_size,
        dataset.len(),
        config.epochs,
    )?;

    Ok((forward, optimizer, metric, metric_budget))
}

/// Trains `module` on `dataset` outputing metrics to `run` with given `config` on `device`.
pub fn module_train(
    binary: Arc<RwLock<BinaryModule>>,
    dataset: Arc<RwLock<Dataset>>,
    run: Arc<RwLock<Run>>,
    config: TrainConfig,
    device: Device,
    model_hash: String,
    dataset_hash: String,
    client_info: Option<ClientInfo>,
    chkpt: Arc<RwLock<CheckPoint>>,
) {
    tokio::spawn(async move {
        let start_time = Instant::now();
        let epochs = config.epochs;
        let batch_size = config.batch_size;
        let per_epoch_checkpoint = config.per_n_epochs_checkpoint;
        let per_n_step_checkpoint = config.per_n_steps_checkpoint;
        let binary = binary.read().unwrap();
        let dataset = dataset.read().unwrap();

        let mut chkpt_guard = chkpt.write().unwrap();

        let (optimizer_state, weights) = chkpt_guard.get_chkpt();
        let mut module: Module = (&*binary).try_into().unwrap();
        module.set_device(device);
        match tcherror_to_status(build_train_context(
            &mut module,
            &dataset,
            config,
            &optimizer_state,
            weights,
        )) {
            Ok((forward, optimizer, metric, metric_budget)) => {
                let trainer = Trainer::new(
                    forward,
                    &dataset,
                    optimizer,
                    metric,
                    metric_budget,
                    device,
                    epochs as usize,
                    batch_size as usize,
                    &mut chkpt_guard,
                    per_epoch_checkpoint,
                    per_n_step_checkpoint,
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
                    match tcherror_to_status(res.map(|(epoch, batch, value, std)| Metric {
                        epoch,
                        batch,
                        value,
                        nb_epochs,
                        nb_batches,
                        uncertainty: 2.0 * std,
                    })) {
                        Ok(m) => *run.write().unwrap() = Run::Ok(m),
                        Err(e) => {
                            *run.write().unwrap() = Run::Error(e);
                            break;
                        }
                    }
                }
                telemetry::add_event(
                    TelemetryEventProps::TrainerLog {
                        log_type: Some("end_training".to_string()),
                        model_hash: Some(model_hash),
                        dataset_hash: Some(dataset_hash),
                        time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    },
                    client_info,
                );
                match &*run.read().unwrap() {
                    Run::Ok(_) => info!(
                    target: "BastionAI",
                                "Model trained successfully in {}ms",
                                start_time.elapsed().as_millis()
                            ),
                    Run::Error(e) => info!(
                    target: "BastionAI",
                                "Model training failed in {}ms: {}",
                                start_time.elapsed().as_millis(),
                                e
                            ),
                    _ => info!(
                    target: "BastionAI",
                                "Model training failed in {}ms",
                                start_time.elapsed().as_millis()
                            ),
                }
            }
            Err(e) => *run.write().unwrap() = Run::Error(e),
        };
    });
}

/// Tests `module` on `dataset` outputing metrics to `run` with given `config` on `device`.
pub fn module_test(
    chkpt: Arc<RwLock<CheckPoint>>,
    binary: Arc<RwLock<BinaryModule>>,
    dataset: Arc<RwLock<Dataset>>,
    run: Arc<RwLock<Run>>,
    config: TestConfig,
    device: Device,
    model_hash: String,
    dataset_hash: String,
    client_info: Option<ClientInfo>,
) {
    tokio::spawn(async move {
        let dataset = dataset.read().unwrap();
        let batch_size = config.batch_size as usize;
        let chkpt = &chkpt.read().unwrap();
        let chkpts_data = &chkpt.data;
        let last_chkpt = &chkpts_data[chkpts_data.len() - 1];

        let loaded_chkpt = Tensor::load_multi_from_stream(Cursor::new(last_chkpt)).unwrap(); // Fix later with more detailed errors.

        let mut module: Module = (&*binary.read().unwrap()).try_into().unwrap(); // Fix later with more detailed errors.

        match tcherror_to_status(build_test_context(
            &mut module,
            &dataset,
            config,
            chkpt.private,
        )) {
            Ok((forward, metric, metric_budget, mut params)) => {
                params.override_parameters(loaded_chkpt).unwrap(); // Fix later with more detailed errors.
                let tester =
                    Tester::new(forward, &dataset, metric, metric_budget, device, batch_size);
                let nb_batches = tester.nb_batches() as i32;

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
                    *run.write().unwrap() =
                        match tcherror_to_status(res.map(|(batch, value, std)| Metric {
                            epoch: 0,
                            batch,
                            value,
                            nb_epochs: 1,
                            nb_batches,
                            uncertainty: 2.0 * std,
                        })) {
                            Ok(m) => Run::Ok(m),
                            Err(e) => Run::Error(e),
                        };
                }
                telemetry::add_event(
                    TelemetryEventProps::TrainerLog {
                        log_type: Some("end_testing".to_string()),
                        model_hash: Some(model_hash),
                        dataset_hash: Some(dataset_hash),
                        time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    },
                    client_info,
                );
                info!(
                target: "BastionAI",
                            "Model tested successfully in {}ms",
                            start_time.elapsed().as_millis()
                        );
            }
            Err(e) => *run.write().unwrap() = Run::Error(e),
        }
    });
}
