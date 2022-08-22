use tch::{Tensor, TchError};
use crate::nn::Parameters;
use super::{initialize_statistics, Optimizer};

/// Adam Optimizer
/// 
/// Updates contained parameters using the Adam algorithm.
/// This is a reimplementation of Pytorch's [Adam] in Rust.
/// 
/// [Adam]: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
pub struct Adam {
    learning_rate: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    weight_decay: f64,
    amsgrad: bool,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    v_hat_max: Vec<Option<Tensor>>,
    t: i32,
    pub parameters: Parameters,
}

impl Adam {
    pub fn new(parameters: Parameters, learning_rate: f64) -> Self {
        Adam {
            learning_rate: learning_rate,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.,
            amsgrad: false,
            m: initialize_statistics(parameters.len()),
            v: initialize_statistics(parameters.len()),
            v_hat_max: initialize_statistics(parameters.len()),
            t: 1,
            parameters,
        }
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

impl Optimizer for Adam {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        self.parameters.zero_grad();
        Ok(())
    }

    fn step(&mut self) -> Result<(), TchError> {
        self.parameters.update(|i, x, mut grad| {
            if self.weight_decay != 0. {
                // grad = grad + weight_decay * x;
                grad = grad.f_add(&x.f_mul_scalar(self.weight_decay)?)?;
            }
            if let Some(m) = &mut self.m[i] {
                // m = beta_1 * m + (1 - beta_1) * grad
                *m = m.f_mul_scalar(self.beta_1)?.f_add(&grad.f_mul_scalar(1. - self.beta_1)?)?;
            } else {
                self.m[i] = Some(grad.f_mul_scalar(1. - self.beta_1)?);
            }
            if let Some(v) = &mut self.v[i] {
                // v = beta_2 * v + (1 - beta_1) * grad ** 2
                *v = v.f_mul_scalar(self.beta_2)?.f_add(&grad.f_square()?.f_mul_scalar(1. - self.beta_2)?)?;
            } else {
                self.v[i] = Some(grad.f_square()?.f_mul_scalar(1. - self.beta_2)?);
            }
            // m_hat = m / (1 - beta_1 ** t)
            let m_hat = self.m[i].as_ref().unwrap().f_div_scalar(1. - self.beta_1.powi(self.t))?;
            // v_hat = v / (1 - beta_2 ** t)
            let v_hat = self.v[i].as_ref().unwrap().f_div_scalar(1. - self.beta_2.powi(self.t))?;

            if self.amsgrad {
                if let Some(v_hat_max) = &mut self.v_hat_max[i] {
                    // v_hat_max = max(v_hat_max, v_hat)
                    *v_hat_max = v_hat_max.f_maximum(&v_hat)?;
                } else {
                    // v_hat_max = v_hat
                    self.v_hat_max[i] = Some(v_hat.f_detach_copy()?);
                }
                // update = learning_rate * m_hat / (sqrt(v_hat_max) + epsilon)
                m_hat.f_div(&self.v_hat_max[i].as_ref().unwrap().f_sqrt()?.f_add_scalar(self.epsilon)?)?.f_mul_scalar(self.learning_rate)
            } else {
                // update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
                m_hat.f_div(&v_hat.f_sqrt()?.f_add_scalar(self.epsilon)?)?.f_mul_scalar(self.learning_rate)
            }
        })
    }
}
