use tch::{Tensor, TchError, Device};
use std::sync::{Arc, RwLock};
use crate::nn::Module;
use crate::data::{Dataset, DatasetIter};
use crate::optim::Optimizer;

fn tensors_to_device(tensors: Vec<Tensor>, device: Device) -> Result<Vec<Tensor>, TchError> {
    let mut tensors_ = Vec::with_capacity(tensors.len());
    for tensor in tensors.iter() {
        tensors_.push(tensor.f_to(device)?);
    }
    Ok(tensors_)
}

pub struct ModuleTrainer {
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    optimizer: Box<dyn Optimizer + Send>,
    metric: Metric,
    device: Device,
    epochs: usize,
    batch_size: usize,
    dataloader: std::iter::Enumerate<DatasetIter>,
    current_epoch: usize,
}

impl ModuleTrainer {
    pub fn new(
        module: Arc<RwLock<Module>>,
        dataset: Arc<RwLock<Dataset>>,
        optimizer: Box<dyn Optimizer + Send>,
        metric: Metric,
        device: Device,
        epochs: usize,
        batch_size: usize,
    ) -> ModuleTrainer {
        ModuleTrainer {
            module,
            dataset: Arc::clone(&dataset),
            optimizer,
            metric,
            device,
            epochs,
            batch_size,
            dataloader: Dataset::iter_shuffle(dataset, batch_size).enumerate(),
            current_epoch: 0,
        }
    }

    pub fn train_on_batch(&mut self, i: usize, inputs: Vec<Tensor>, labels: Tensor) -> Result<(i32, i32, f32), TchError> {
        let inputs = tensors_to_device(inputs, self.device)?;
        let labels = labels.f_to(self.device)?;
        let outputs = self.module.read().unwrap().forward(&inputs)?;
        let loss = self.metric.compute(&outputs, &labels)?;
        self.optimizer.zero_grad()?;
        loss.backward();
        self.optimizer.step()?;
        Ok((self.current_epoch as i32, i as i32, self.metric.value()))
    }

    pub fn nb_epochs(&self) -> usize {
        self.epochs
    }

    pub fn nb_batches(&self) -> usize {
        self.dataset.read().unwrap().len() / self.batch_size
    }
}

impl Iterator for ModuleTrainer {
    type Item = Result<(i32, i32, f32), TchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((i, (inputs, labels))) = self.dataloader.next() {
            Some(self.train_on_batch(i, inputs, labels))
        } else {
            self.current_epoch += 1;
            self.metric.reset();
            if self.current_epoch < self.epochs {
                self.dataloader =
                    Dataset::iter_shuffle(Arc::clone(&self.dataset), self.batch_size).enumerate();
                self.next()
            } else {
                None
            }
        }
    }
}

pub struct ModuleTester {
    module: Arc<RwLock<Module>>,
    metric: Metric,
    device: Device,
    dataloader: std::iter::Enumerate<DatasetIter>,
    nb_batches: usize,
}

impl ModuleTester {
    pub fn new(
        module: Arc<RwLock<Module>>,
        dataset: Arc<RwLock<Dataset>>,
        metric: Metric,
        device: Device,
        batch_size: usize,
    ) -> ModuleTester {
        let nb_batches = dataset.read().unwrap().len() / batch_size;
        ModuleTester {
            module,
            metric,
            device,
            dataloader: Dataset::iter_shuffle(dataset, batch_size).enumerate(),
            nb_batches,
        }
    }

    pub fn test_on_batch(&mut self, i: usize, inputs: Vec<Tensor>, labels: Tensor) -> Result<(i32, f32), TchError> {
        let inputs = tensors_to_device(inputs, self.device)?;
        let labels = labels.f_to(self.device)?;
        let outputs = self.module.read().unwrap().forward(&inputs)?;
        let _ = self.metric.compute(&outputs, &labels)?;
        Ok((i as i32, self.metric.value()))
    }

    pub fn nb_batches(&self) -> usize {
        self.nb_batches
    }
}

impl Iterator for ModuleTester {
    type Item = Result<(i32, f32), TchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((i, (inputs, labels))) = self.dataloader.next() {
            Some(self.test_on_batch(i, inputs, labels))
        } else {
            None
        }
    }
}

/// A loss function with average statistics
pub struct Metric {
    loss_fn: Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor, TchError> + Send>,
    value: f32,
    nb_samples: usize,
}

impl Metric {
    /// Returns a `Metric` corresponding to given name, if not available raises an error.
    pub fn try_from_name(loss_name: &str) -> Result<Self, TchError> {
        Ok(Metric {
            loss_fn: match loss_name {
                "accuracy" => Box::new(|output, label| {
                    let prediction = output.f_argmax(-1, false)?.double_value(&[]);
                    Ok(if prediction == label.double_value(&[]) {
                        Tensor::of_slice(&[1]).f_view([])?
                    } else {
                        Tensor::of_slice(&[1]).f_view([])?
                    })
                }),
                "l2" => Box::new(|output, label| output.f_mse_loss(label, tch::Reduction::Mean)),
                "cross_entropy" => Box::new(|output, label| output.f_cross_entropy_loss::<Tensor>(label, None, tch::Reduction::Mean, -100, 0.)),
                s => return Err(TchError::FileFormat(String::from(format!("Invalid loss name, unknown loss {}.", s)))),
            },
            value: 0.0,
            nb_samples: 1,
        })
    }

    /// Computes the metric's value given `output` and `label` and updates the average.
    pub fn compute(&mut self, output: &Tensor, label: &Tensor) -> Result<Tensor, TchError> {
        let loss = (self.loss_fn)(output, label)?;
        self.value += 1. / (self.nb_samples as f32) * (loss.double_value(&[]) as f32 - self.value);
        self.nb_samples += 1;
        Ok(loss)
    }

    /// Returns the average.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Resets the metric
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.nb_samples = 1;
    }
}

