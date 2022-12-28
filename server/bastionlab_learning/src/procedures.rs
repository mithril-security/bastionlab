use crate::data::privacy_guard::{PrivacyBudget, PrivacyGuard};
use crate::data::{Dataset, DatasetIter};
use crate::nn::{CheckPoint, Forward};
use crate::optim::Optimizer;
use tch::{Device, Kind, TchError, Tensor};

fn inputs_to_device(
    inputs: Vec<PrivacyGuard<Tensor>>,
    device: Device,
) -> Result<Vec<PrivacyGuard<Tensor>>, TchError> {
    let mut inputs_ = Vec::with_capacity(inputs.len());
    for tensor in inputs.iter() {
        inputs_.push(tensor.f_to(device)?);
    }
    Ok(inputs_)
}

/// A basic parametrizable loop for training a model.
///
/// This struct implements [`std::Iter::Iterator`] and yields
/// a metric value for every step (i.e. every batch of every epoch).
pub struct Trainer<'a> {
    forward: Forward<'a>,
    dataset: &'a Dataset,
    optimizer: Box<dyn Optimizer + 'a>,
    metric: Metric,
    metric_budget: PrivacyBudget,
    device: Device,
    epochs: usize,
    batch_size: usize,
    dataloader: std::iter::Enumerate<DatasetIter<'a>>,
    current_epoch: usize,
    chkpt: &'a mut CheckPoint,
    per_n_epochs_chkpt: i32,
    per_n_steps_chkpt: i32,
}

impl<'a> Trainer<'a> {
    pub fn new(
        forward: Forward<'a>,
        dataset: &'a Dataset,
        optimizer: Box<dyn Optimizer + 'a>,
        metric: Metric,
        metric_budget: PrivacyBudget,
        device: Device,
        epochs: usize,
        batch_size: usize,
        chkpt: &'a mut CheckPoint,
        per_n_epochs_chkpt: i32,
        per_n_steps_chkpt: i32,
    ) -> Trainer<'a> {
        Trainer {
            forward,
            dataset,
            optimizer,
            metric,
            metric_budget,
            device,
            epochs,
            batch_size,
            dataloader: dataset.iter_shuffle(batch_size).enumerate(),
            current_epoch: 0,
            chkpt,
            per_n_epochs_chkpt,
            per_n_steps_chkpt,
        }
    }

    pub fn train_on_batch(
        &mut self,
        i: usize,
        inputs: Vec<PrivacyGuard<Tensor>>,
        labels: PrivacyGuard<Tensor>,
    ) -> Result<(i32, i32, f32, f32), TchError> {
        let inputs = inputs_to_device(inputs, self.device)?;
        let labels = labels.f_to(self.device)?;
        let outputs = self.forward.forward(inputs)?;
        let loss = self.metric.compute(&outputs, &labels)?;
        self.optimizer.zero_grad()?;
        loss.backward();
        self.optimizer.step()?;
        let (value, std) = self.metric.value(self.metric_budget)?;
        Ok((self.current_epoch as i32, i as i32, value, std))
    }

    pub fn nb_epochs(&self) -> usize {
        self.epochs
    }

    pub fn nb_batches(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    fn checkpoint(&mut self) -> Result<(), TchError> {
        let params = self.optimizer.into_bytes()?; // Fix later with more detailed errors.
        let optim_state = self.optimizer.get_state()?;
        self.chkpt.log_chkpt(&params, optim_state)?;
        Ok(())
    }
}

impl<'a> Iterator for Trainer<'a> {
    type Item = Result<(i32, i32, f32, f32), TchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((i, (inputs, labels))) = self.dataloader.next() {
            let v = Some(self.train_on_batch(i, inputs, labels));

            // Per n-step checkpointing.
            if self.per_n_steps_chkpt > 0 && i % self.per_n_steps_chkpt as usize == 0 {
                self.checkpoint().unwrap()
            }
            v
        } else {
            self.current_epoch += 1;
            self.metric.reset();
            if self.current_epoch < self.epochs {
                self.dataloader = self.dataset.iter_shuffle(self.batch_size).enumerate();
                let v = self.next();

                // Per n-epoch checkpointing.
                if self.per_n_epochs_chkpt > 0
                    && self.current_epoch % self.per_n_epochs_chkpt as usize == 0
                {
                    self.checkpoint().unwrap()
                }
                v
            } else {
                // Default checkpointing.
                if self.per_n_epochs_chkpt == 0 && self.per_n_steps_chkpt == 0 {
                    self.checkpoint().unwrap()
                }
                None
            }
        }
    }
}

/// A basic parametriazable loop for testing a model.
///
/// This struct implements [`std::Iter::Iterator`] and yields
/// a metric value for every step (i.e. every batch).
pub struct Tester<'a> {
    forward: Forward<'a>,
    metric: Metric,
    metric_budget: PrivacyBudget,
    device: Device,
    dataloader: std::iter::Enumerate<DatasetIter<'a>>,
    nb_batches: usize,
}

impl<'a> Tester<'a> {
    pub fn new(
        forward: Forward<'a>,
        dataset: &'a Dataset,
        metric: Metric,
        metric_budget: PrivacyBudget,
        device: Device,
        batch_size: usize,
    ) -> Tester<'a> {
        let nb_batches = dataset.len() / batch_size;
        Tester {
            forward,
            metric,
            metric_budget,
            device,
            dataloader: dataset.iter_shuffle(batch_size).enumerate(),
            nb_batches,
        }
    }

    pub fn test_on_batch(
        &mut self,
        i: usize,
        inputs: Vec<PrivacyGuard<Tensor>>,
        labels: PrivacyGuard<Tensor>,
    ) -> Result<(i32, f32, f32), TchError> {
        let inputs = inputs_to_device(inputs, self.device)?;
        let labels = labels.f_to(self.device)?;
        let outputs = self.forward.forward(inputs)?;
        let _ = self.metric.compute(&outputs, &labels)?;
        let (value, std) = self.metric.value(self.metric_budget)?;
        Ok((i as i32, value, std))
    }

    pub fn nb_batches(&self) -> usize {
        self.nb_batches
    }
}

impl<'a> Iterator for Tester<'a> {
    type Item = Result<(i32, f32, f32), TchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((i, (inputs, labels))) = self.dataloader.next() {
            Some(self.test_on_batch(i, inputs, labels))
        } else {
            None
        }
    }
}

/// A loss (or metric) function with average statistics
pub struct Metric {
    loss_fn: Box<
        dyn Fn(
                &PrivacyGuard<Tensor>,
                &PrivacyGuard<Tensor>,
            ) -> Result<(PrivacyGuard<Tensor>, PrivacyGuard<Tensor>), TchError>
            + Send,
    >,
    value: Option<PrivacyGuard<Tensor>>,
    clipping: (f64, f64),
    nb_samples: usize,
}

impl Metric {
    /// Returns a `Metric` corresponding to given name, if not available raises an error.
    pub fn try_from_name(loss_name: &str) -> Result<Self, TchError> {
        let (loss_fn, clipping): (
            Box<
                dyn Fn(
                        &PrivacyGuard<Tensor>,
                        &PrivacyGuard<Tensor>,
                    )
                        -> Result<(PrivacyGuard<Tensor>, PrivacyGuard<Tensor>), TchError>
                    + Send,
            >,
            (f64, f64),
        ) = match loss_name {
            "accuracy" => (
                Box::new(|output, label| {
                    let prediction = output
                        .f_argmax(-1, false)?
                        .f_sub(label)?
                        .f_abs()?
                        .f_clamp(0.0, 1.0)?
                        .f_sum(Kind::Float)?
                        .f_mul_scalar(-1.0 / label.batch_size()? as f64)?
                        .f_add_scalar(1.0)?;
                    Ok((prediction.f_clone()?, prediction))
                }),
                (0.0, 1.0),
            ),
            "l2" => (
                Box::new(|output, label| {
                    output.f_mse_loss(label, (0.0, 10.0), tch::Reduction::Mean)
                }),
                (0.0, 10.0),
            ),
            "cross_entropy" => (
                Box::new(|output, label| {
                    let weight: Option<Tensor> = None;
                    output.f_cross_entropy_loss(
                        label,
                        (0.0, 10.0),
                        weight,
                        tch::Reduction::Mean,
                        -100,
                        0.,
                    )
                }),
                (0.0, 10.0),
            ),
            s => {
                return Err(TchError::FileFormat(String::from(format!(
                    "Invalid loss name, unknown loss {}.",
                    s
                ))))
            }
        };
        Ok(Metric {
            loss_fn,
            value: None,
            clipping,
            nb_samples: 0,
        })
    }

    /// Computes the metric's value given `output` and `label` and updates the average.
    pub fn compute(
        &mut self,
        output: &PrivacyGuard<Tensor>,
        label: &PrivacyGuard<Tensor>,
    ) -> Result<PrivacyGuard<Tensor>, TchError> {
        let expansion = output.batch_size()? / label.batch_size()?;
        let loss = if expansion != 1 {
            (self.loss_fn)(output, &label.expand_batch_dim(expansion)?)?
        } else {
            (self.loss_fn)(output, label)?
        };
        // let loss = (self.loss_fn)(output, label)?;
        let detached_loss = loss.1.f_clone()?;
        self.value = match &self.value {
            Some(v) => Some(
                v.f_mul_scalar(self.nb_samples as f64 / (self.nb_samples + 1) as f64)?
                    .f_add(&detached_loss.f_mul_scalar(1.0 / self.nb_samples as f64)?)?,
            ),
            None => Some(detached_loss),
        };
        self.nb_samples += 1;
        Ok(loss.0)
    }

    /// Privately returns the average with given budget.
    pub fn value(&self, budget: PrivacyBudget) -> Result<(f32, f32), TchError> {
        match &self.value {
            Some(x) => {
                let (value, std) = x.f_clone()?.get_private_with_std(budget)?;
                Ok((
                    (value.f_double_value(&[])? as f32)
                        .clamp(self.clipping.0 as f32, self.clipping.1 as f32),
                    std,
                ))
            }
            None => Ok((0.0, 0.0)),
        }
    }

    /// Resets the metric
    pub fn reset(&mut self) {
        self.value = None;
        self.nb_samples = 1;
    }
}
