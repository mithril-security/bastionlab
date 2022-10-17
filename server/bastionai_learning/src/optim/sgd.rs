use std::collections::HashMap;

use super::{
    bytes_to_stats, initialize_statistics, optimizer::OptimizerStateType, stats_to_bytes, Optimizer,
};
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
    statistics: HashMap<String, Option<Tensor>>,
    pub parameters: Parameters<'a>,
}

impl<'a> SGD<'a> {
    /// Returns a new SGD optimizer to update given `parameters` using given `learning_rate`.
    pub fn new(parameters: Parameters<'a>, learning_rate: f64) -> Self {
        SGD {
            learning_rate,
            weight_decay: 0.,
            momentum: 0.,
            dampening: 0.,
            nesterov: false,
            statistics: initialize_statistics(),
            parameters,
        }
    }
    /// Restores an Optimizer to the latest training checkpoint with `optimizer_state` and
    pub fn load_from_checkpoint(
        optimizer_state: &Option<OptimizerStateType>,
        weights: &[u8],
        learning_rate: f64,
        mut parameters: Parameters<'a>,
    ) -> Result<Self, TchError> {
        let statistics = match optimizer_state {
            Some(v) => match v {
                OptimizerStateType::SGD { statistics } => bytes_to_stats(&statistics[..])?,
                _ => initialize_statistics(),
            },
            None => initialize_statistics(),
        };
        let weights = bytes_to_stats(weights)?;
        let weights = weights
            .into_iter()
            .map(|(k, v)| (k.clone(), v.unwrap()))
            .collect::<Vec<_>>();
        parameters.override_parameters(weights)?;
        Ok(SGD {
            learning_rate,
            weight_decay: 0.,
            momentum: 0.,
            dampening: 0.,
            nesterov: false,
            statistics,
            parameters,
        })
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
        self.parameters.update(|name, x, mut grad| {
            if self.weight_decay != 0. {
                // grad = grad + weight_decay * x
                grad = grad.f_add(&x.f_mul_scalar(self.weight_decay)?)?;
            }
            if self.momentum != 0. {
                match self.statistics.get_mut(name) {
                    Some(b) => {
                        if let Some(b) = b {
                            // b = momentum * b + (1 - dampening) * grad
                            *b = b
                                .f_mul_scalar(self.momentum)?
                                .f_add(&grad.f_mul_scalar(1. - self.dampening)?)?;
                        }
                    }
                    None => {
                        self.statistics
                            .insert(name.clone(), Some(grad.f_detach_copy()?));
                    }
                }
                if self.nesterov {
                    // grad = grad + momentum * statistics
                    let other = self
                        .statistics
                        .get(name)
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .f_mul_scalar(self.momentum)?;
                    grad = grad.f_add(&other)?;
                } else {
                    grad = (self.statistics.get(name).unwrap())
                        .as_ref()
                        .unwrap()
                        .f_detach_copy()?;
                }
            }
            // update = learning_rate * grad
            grad.f_mul_scalar(self.learning_rate)
        })
    }

    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError> {
        self.parameters.into_bytes()
    }
    fn get_state(&mut self) -> Result<OptimizerStateType, TchError> {
        let statistics = stats_to_bytes(&self.statistics)?;
        Ok(OptimizerStateType::SGD { statistics })
    }
}
