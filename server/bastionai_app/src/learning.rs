use crate::remote_torch::{train_config, Metric, TestConfig, TrainConfig};
use crate::utils::tcherror_to_status;
use bastionai_learning::data::privacy_guard::PrivacyBudget;
use bastionai_learning::data::Dataset;
use bastionai_learning::nn::{Forward, LossType, Module};
use bastionai_learning::optim::{Adam, Optimizer, SGD};
use bastionai_learning::procedures::{self, Tester, Trainer};
use std::ops::DerefMut;
use std::sync::{Arc, RwLock};
use tch::{Device, TchError};
use tonic::Status;

#[derive(Debug)]
pub enum Run {
    Ok(Metric),
    Error(Status),
    Pending,
}

fn build_shared_context(
    metric: &str,
    metric_eps: f32,
    batch_size: i32,
    dataset_size: usize,
    nb_epochs: i32,
) -> Result<(procedures::Metric, PrivacyBudget), TchError> {
    let total_nb_batches = dataset_size as i32 / batch_size * nb_epochs;
    println!("total batches: {}, dataset size: {}, batch_size: {}, epochs: {}", total_nb_batches, dataset_size, batch_size, nb_epochs);
    let metric = procedures::Metric::try_from_name(metric)?;
    let metric_budget = if metric_eps < 0.0 {
        PrivacyBudget::NotPrivate
    } else {
        PrivacyBudget::Private(metric_eps / total_nb_batches as f32)
    };

    Ok((metric, metric_budget))
}

fn build_test_context<'a>(
    module: &'a Module,
    dataset: &Dataset,
    config: TestConfig,
) -> Result<(Forward<'a>, procedures::Metric, PrivacyBudget), TchError> {
    let forward = module.forward_fn();
    let (metric, metric_budget) = build_shared_context(
        &config.metric,
        config.metric_eps,
        config.batch_size,
        dataset.len(),
        1,
    )?;

    println!("metric_budget: {:?}", metric_budget);
    Ok((forward, metric, metric_budget))
}

fn build_train_context<'a>(
    module: &'a mut Module,
    dataset: &Dataset,
    config: TrainConfig,
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

    let (metric, metric_budget) = build_shared_context(
        &config.metric,
        config.metric_eps,
        config.batch_size,
        dataset.len(),
        config.epochs,
    )?;

    println!("budget: {:?}, metric_budget: {:?}", config.eps / (q * t.sqrt()), metric_budget);
    Ok((forward, optimizer, metric, metric_budget))
}

pub fn module_train(
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    run: Arc<RwLock<Run>>,
    config: TrainConfig,
    device: Device,
) {
    tokio::spawn(async move {
        let epochs = config.epochs;
        let batch_size = config.batch_size;
        let mut module = module.write().unwrap();
        let dataset = dataset.read().unwrap();
        match tcherror_to_status(build_train_context(&mut module, &dataset, config)) {
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
                );
                let nb_epochs = trainer.nb_epochs() as i32;
                let nb_batches = trainer.nb_batches() as i32;
                for res in trainer {
                    *run.write().unwrap() =
                        match tcherror_to_status(res.map(|(epoch, batch, value, std)| Metric {
                            epoch,
                            batch,
                            value,
                            nb_epochs,
                            nb_batches,
                            uncertainty: 2.0 * std,
                        })) {
                            Ok(m) => Run::Ok(m),
                            Err(e) => Run::Error(e),
                        };
                }
            }
            Err(e) => *run.write().unwrap() = Run::Error(e),
        };
    });
}

pub fn module_test(
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    run: Arc<RwLock<Run>>,
    config: TestConfig,
    device: Device,
) {
    tokio::spawn(async move {
        let module = module.write().unwrap();
        let dataset = dataset.read().unwrap();
        let batch_size = config.batch_size as usize;
        match tcherror_to_status(build_test_context(&module, &dataset, config)) {
            Ok((forward, metric, metric_budget)) => {
                let tester =
                    Tester::new(forward, &dataset, metric, metric_budget, device, batch_size);
                let nb_batches = tester.nb_batches() as i32;
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
            }
            Err(e) => *run.write().unwrap() = Run::Error(e),
        }
    });
}
