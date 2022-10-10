use super::{initialize_statistics, Optimizer};
use crate::nn::Parameters;
use tch::{TchError, Tensor};

/// Stochastic Gradient Descent Optimizer
///
/// Updates contained parameters using the SGD algorithm.
/// This optimizer also supports weight decay, momentum, dampening
/// and nesterov updates.
///
/// It is a reimplementation of Pytorch's [SGD] in Rust.
///
/// [SGD]: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
pub struct SGD<'a> {
    learning_rate: f64,
    weight_decay: f64,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    statistics: Vec<Option<Tensor>>,
    pub parameters: Parameters<'a>,
}

impl<'a> SGD<'a> {
    /// Returns a new SGD optimizer to update given `parameters` using given `learning_rate`.
    pub fn new(parameters: Parameters<'a>, learning_rate: f64) -> Self {
        SGD {
            learning_rate: learning_rate,
            weight_decay: 0.,
            momentum: 0.,
            dampening: 0.,
            nesterov: false,
            statistics: initialize_statistics(parameters.len()),
            parameters,
        }
    }
    /// Sets weight_decay.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    /// Sets momentum.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
    /// Sets dampening factor.
    pub fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }
    /// Enables or disables nesterov updates.
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl<'a> Optimizer for SGD<'a> {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        self.parameters.zero_grad();
        Ok(())
    }

    fn step(&mut self) -> Result<(), TchError> {
        self.parameters.update(|i, x, mut grad| {
            if self.weight_decay != 0. {
                // grad = grad + weight_decay * x
                grad = grad.f_add(&x.f_mul_scalar(self.weight_decay)?)?;
            }
            if self.momentum != 0. {
                if let Some(b) = &mut self.statistics[i] {
                    // b = momentum * b + (1 - dampening) * grad
                    *b = b
                        .f_mul_scalar(self.momentum)?
                        .f_add(&grad.f_mul_scalar(1. - self.dampening)?)?;
                } else {
                    self.statistics[i] = Some(grad.f_detach_copy()?)
                }
                if self.nesterov {
                    // grad = grad + momentum * statistics
                    grad = grad.f_add(
                        &(&self.statistics[i])
                            .as_ref()
                            .unwrap()
                            .f_mul_scalar(self.momentum)?,
                    )?;
                } else {
                    grad = (&self.statistics[i]).as_ref().unwrap().f_detach_copy()?;
                }
            }
            // update = learning_rate * grad
            grad.f_mul_scalar(self.learning_rate)
        })
    }

    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError> {
        self.parameters.into_bytes()
    }
}
