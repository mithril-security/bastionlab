use std::collections::HashMap;

use super::{
    bytes_to_stats, initialize_statistics, optimizer::OptimizerStateType, stats_to_bytes, Optimizer,
};
use crate::nn::Parameters;
use tch::{TchError, Tensor};

/// Adam Optimizer
///
/// Updates contained parameters using the Adam algorithm.
/// This is a reimplementation of Pytorch's [Adam] in Rust.
///
/// [Adam]: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
pub struct Adam<'a> {
    learning_rate: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    weight_decay: f64,
    amsgrad: bool,
    m: HashMap<String, Option<Tensor>>,
    v: HashMap<String, Option<Tensor>>,
    v_hat_max: HashMap<String, Option<Tensor>>,
    t: i32,
    pub parameters: Parameters<'a>,
}

impl<'a> Adam<'a> {
    pub fn new(parameters: Parameters<'a>, learning_rate: f64) -> Self {
        Adam {
            learning_rate,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.,
            amsgrad: false,
            m: initialize_statistics(),
            v: initialize_statistics(),
            v_hat_max: initialize_statistics(),
            t: 1,
            parameters,
        }
    }
    /// Restores an Optimizer to the latest training checkpoint with `optimizer_state` and
    pub fn load_from_checkpoint(
        optimizer_state: &'a Option<OptimizerStateType>,
        weights: &[u8],
        learning_rate: f64,
        mut parameters: Parameters<'a>,
    ) -> Result<Self, TchError> {
        let empty_stats = (
            initialize_statistics(),
            initialize_statistics(),
            initialize_statistics(),
            1,
        );
        let (m, v, v_hat_max, t) = match optimizer_state {
            Some(state) => match state {
                OptimizerStateType::Adam { m, v, v_hat_max, t } => (
                    bytes_to_stats(m)?,
                    bytes_to_stats(v)?,
                    bytes_to_stats(v_hat_max)?,
                    *t,
                ),
                _ => empty_stats,
            },
            None => empty_stats,
        };
        let weights = bytes_to_stats(weights)?;
        let weights = weights
            .into_iter()
            .map(|(k, v)| (k.clone(), v.unwrap()))
            .collect::<Vec<_>>();
        parameters.override_parameters(weights)?;
        Ok(Adam {
            learning_rate,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.,
            amsgrad: false,
            m,
            v,
            v_hat_max,
            t,
            parameters,
        })
    }
    pub fn beta_1(mut self, beta_1: f64) -> Self {
        self.beta_1 = beta_1;
        self
    }
    pub fn beta_2(mut self, beta_2: f64) -> Self {
        self.beta_2 = beta_2;
        self
    }
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl<'a> Optimizer for Adam<'a> {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        self.parameters.zero_grad();
        Ok(())
    }

    fn step(&mut self) -> Result<(), TchError> {
        self.parameters.update(|name, x, mut grad| {
            if self.weight_decay != 0. {
                // grad = grad + weight_decay * x;
                grad = grad.f_add(&x.f_mul_scalar(self.weight_decay)?)?;
            }
            match self.m.get_mut(name) {
                Some(m) => {
                    if let Some(m) = m {
                        // m = beta_1 * m + (1 - beta_1) * grad
                        *m = m
                            .f_mul_scalar(self.beta_1)?
                            .f_add(&grad.f_mul_scalar(1. - self.beta_1)?)?;
                    }
                }
                None => {
                    self.m
                        .insert(name.clone(), Some(grad.f_mul_scalar(1. - self.beta_1)?));
                }
            }

            match self.v.get_mut(name) {
                Some(v) => {
                    if let Some(v) = v {
                        // v = beta_2 * v + (1 - beta_1) * grad ** 2
                        *v = v
                            .f_mul_scalar(self.beta_2)?
                            .f_add(&grad.f_square()?.f_mul_scalar(1. - self.beta_2)?)?;
                    }
                }
                None => {
                    self.v.insert(
                        name.clone(),
                        Some(grad.f_square()?.f_mul_scalar(1. - self.beta_2)?),
                    );
                }
            }

            // m_hat = m / (1 - beta_1 ** t)
            let m_hat = self
                .m
                .get(name)
                .unwrap()
                .as_ref()
                .unwrap()
                .f_div_scalar(1. - self.beta_1.powi(self.t))?;
            // v_hat = v / (1 - beta_2 ** t)
            let v_hat = self
                .v
                .get(name)
                .unwrap()
                .as_ref()
                .unwrap()
                .f_div_scalar(1. - self.beta_2.powi(self.t))?;

            if self.amsgrad {
                match self.v_hat_max.get_mut(name) {
                    Some(v_hat_max) => {
                        if let Some(v_hat_max) = v_hat_max {
                            // v_hat_max = max(v_hat_max, v_hat)
                            *v_hat_max = v_hat_max.f_maximum(&v_hat)?;
                        }
                    }
                    None => {
                        {
                            // v_hat_max = v_hat
                            self.v_hat_max
                                .insert(name.clone(), Some(v_hat.f_detach_copy()?));
                        }
                    }
                }
                // update = learning_rate * m_hat / (sqrt(v_hat_max) + epsilon)
                m_hat
                    .f_div(
                        &self
                            .v_hat_max
                            .get(name)
                            .unwrap()
                            .as_ref()
                            .unwrap()
                            .f_sqrt()?
                            .f_add_scalar(self.epsilon)?,
                    )?
                    .f_mul_scalar(self.learning_rate)
            } else {
                // update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
                m_hat
                    .f_div(&v_hat.f_sqrt()?.f_add_scalar(self.epsilon)?)?
                    .f_mul_scalar(self.learning_rate)
            }
        })
    }

    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError> {
        self.parameters.into_bytes()
    }

    fn get_state(&mut self) -> Result<OptimizerStateType, TchError> {
        let m = stats_to_bytes(&self.m)?;
        let v = stats_to_bytes(&self.v)?;
        let v_hat_max = stats_to_bytes(&self.v_hat_max)?;
        Ok(OptimizerStateType::Adam {
            m,
            v,
            v_hat_max,
            t: self.t,
        })
    }
}
