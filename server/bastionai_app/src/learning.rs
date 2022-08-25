use crate::remote_torch::{Metric, TestConfig, TrainConfig, train_config};
use crate::utils::tcherror_to_status;
use std::ops::DerefMut;
use std::sync::{Arc, RwLock};
use bastionai_learning::data::privacy_guard::PrivacyBudget;
use tch::{Device, TchError};
use tonic::Status;
use bastionai_learning::data::Dataset;
use bastionai_learning::nn::{Module, LossType, Forward};
use bastionai_learning::optim::{Adam, SGD, Optimizer};
use bastionai_learning::procedures::{self, Trainer, Tester};

#[derive(Debug)]
pub enum Run {
    Ok(Metric),
    Error(Status),
    Pending,
}

fn build_train_context<'a>(module: &'a mut Module, config: TrainConfig,) -> Result<(Forward<'a>, Box<dyn Optimizer + 'a>, procedures::Metric), TchError> {
    let (forward, parameters) = match config
            .privacy
            .ok_or(TchError::FileFormat(String::from("Invalid privacy option")))?
        {
            train_config::Privacy::Standard(_) => module.parameters(),
            train_config::Privacy::DifferentialPrivacy(train_config::DpParameters {
                max_grad_norm,
                noise_multiplier,
            }) => module.private_parameters(
                max_grad_norm,
                noise_multiplier,
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
            ) as Box<dyn Optimizer + 'a>,
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
            ) as Box<dyn Optimizer + 'a>,
        };
        
        let metric = procedures::Metric::try_from_name(&config.metric)?;

        Ok((forward, optimizer, metric))
}

pub async fn module_train(
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    run: Arc<RwLock<Run>>,
    config: TrainConfig,
    device: Device,
) -> Result<(), Status> {
    tokio::spawn(async move {
        let epochs = config.epochs;
        let batch_size = config.batch_size;
        let mut module = module.write().unwrap();
        let dataset = dataset.read().unwrap();
        let module_ref = module.deref_mut();
        match tcherror_to_status(build_train_context(module_ref, config)) {
            Ok((forward, optimizer, metric)) => {
                let trainer = Trainer::new(forward, &dataset, optimizer, metric, PrivacyBudget::Private(0.0), device, epochs as usize, batch_size as usize);
                let nb_epochs = trainer.nb_epochs() as i32;
                let nb_batches = trainer.nb_batches() as i32;
                for res in trainer {
                    *run.write().unwrap() = match tcherror_to_status(res.map(|(epoch, batch, value)| Metric {
                        epoch,
                        batch,
                        value,
                        nb_epochs,
                        nb_batches,
                    })) {
                        Ok(m) => Run::Ok(m),
                        Err(e) => Run::Error(e),
                    };
                }
            }
            Err(e) => *run.write().unwrap() = Run::Error(e),
        };
    });

    Ok(())
}

pub async fn module_test(
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    run: Arc<RwLock<Run>>,
    config: TestConfig,
    device: Device,
) -> Result<(), Status> {
    tokio::spawn(async move {
        let module = module.write().unwrap();
        let dataset = dataset.read().unwrap();
        let metric = tcherror_to_status(procedures::Metric::try_from_name(&config.metric));
        match metric {
            Ok(metric) => {
                let tester = Tester::new(module.forward_fn(), &dataset, metric, PrivacyBudget::Private(0.0), device, config.batch_size as usize);
                let nb_batches = tester.nb_batches() as i32;
                for res in tester {
                    *run.write().unwrap() = match tcherror_to_status(res.map(|(batch, value)| Metric {
                        epoch: 0,
                        batch,
                        value,
                        nb_epochs: 1,
                        nb_batches,
                    })) {
                        Ok(m) => Run::Ok(m),
                        Err(e) => Run::Error(e),
                    };
                }
            }
            Err(e) => *run.write().unwrap() = Run::Error(e),
        }
    });

    Ok(())
}
