use tch::{TchError, COptimizer, Tensor, Kind, nn::VarStore, IndexOp};

#[cfg(test)]
mod tests {
    use tch::{TrainableCModule, Device, Tensor, Kind, TchError};
    use tch::nn::VarStore;

    use crate::{Parameters, PrivateParameters, LossType, SGD, Optimizer, l2_loss};

    #[test]
    fn basic_sgd() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/tests/lreg.pt", vs.root()).unwrap();
        let parameters = Parameters::from(vs);
        let mut optimizer = SGD::new(parameters, 0.1);

        let data = vec![
            Tensor::of_slice::<f32>(&[0.]),
            Tensor::of_slice::<f32>(&[1.]),
        ];
        let target = vec![
            Tensor::of_slice::<f32>(&[0.]),
            Tensor::of_slice::<f32>(&[2.]),
        ];
        
        for _ in 0..100 {
            for (x, t) in data.iter().zip(target.iter()) {
                let y = model.forward_ts(&[x]).unwrap();
                let loss = (y - t).abs();
                optimizer.zero_grad().unwrap();
                loss.backward();
                optimizer.step().unwrap();
            }
        }
        let w = &optimizer.parameters.into_inner()[0];
        assert!((w - Tensor::of_slice::<f32>(&[2.])).abs().double_value(&[]) < 0.1);
    }

    #[test]
    fn private_sgd() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/tests/private_lreg.pt", vs.root()).unwrap();
        let parameters = PrivateParameters::new(&vs, 1.0, 0.1, LossType::Mean(2));
        let mut optimizer = SGD::new(parameters, 0.1);

        let data = vec![
            Tensor::of_slice::<f32>(&[0.0, 1.0]).f_view([2, 1]).unwrap(),
            Tensor::of_slice::<f32>(&[0.5, 0.2]).f_view([2, 1]).unwrap(),
        ];
        let target = vec![
            Tensor::of_slice::<f32>(&[0.0, 2.0]).f_view([2, 1]).unwrap(),
            Tensor::of_slice::<f32>(&[1.0, 0.4]).f_view([2, 1]).unwrap(),
        ];
        
        for _ in 0..100 {
            for (x, t) in data.iter().zip(target.iter()) {
                let y = model.forward_ts(&[x]).unwrap();
                let loss = l2_loss(&y, &t).unwrap();
                // y.print();
                optimizer.zero_grad().unwrap();
                loss.backward();
                optimizer.step().unwrap();
            }
        }
        let w = &optimizer.parameters.into_inner()[0];
        assert!(l2_loss(w, &Tensor::of_slice::<f32>(&[2.])).unwrap().double_value(&[]) < 0.1);
    }

}

pub fn l2_loss(output: &Tensor, target: &Tensor) -> Result<Tensor, TchError> {
    output.f_sub(&target)?.f_norm_scalaropt_dim(2, &[1], false)?.f_mean(Kind::Float)
}

pub trait Optimizer {
    fn zero_grad(&mut self) -> Result<(), TchError>;
    fn step(&mut self) -> Result<(), TchError>;
}

impl Optimizer for COptimizer {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        COptimizer::zero_grad(self)
    }
    fn step(&mut self) -> Result<(), TchError> {
        COptimizer::step(self)
    }
}

pub enum Parameters {
    Standard(Vec<Tensor>),
    Private {
        parameters: Vec<Tensor>,
        max_grad_norm: f64,
        noise_multiplier: f64,
        loss_type: LossType,
    },
}

impl Parameters {
    pub fn standard(vs: &VarStore) -> Self {
        Parameters::Standard(vs.trainable_variables())
    }

    pub fn private(vs: &VarStore, max_grad_norm: f64, noise_multiplier: f64, loss_type: LossType) -> Self {
        Parameters::Private { parameters: vs.trainable_variables(), max_grad_norm, noise_multiplier, loss_type }
    }

    pub fn into_inner(mut self) -> Vec<Tensor> {
        self.zero_grad();
        match self {
            Parameters::Standard(parameters) => parameters,
            Parameters::Private { parameters, .. } => parameters,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Parameters::Standard(parameters) => parameters.len(),
            Parameters::Private { parameters, .. } => parameters.len(),
        }
    }

    pub fn zero_grad(&mut self) {
        match self {
            Parameters::Standard(parameters) => {
                for param in parameters.iter_mut() {
                    param.zero_grad();
                }
            }
            Parameters::Private { parameters, .. } => {
                for param in parameters.iter_mut() {
                    param.zero_grad();
                }
            }
        }
        
    }

    pub fn update(&mut self, mut update_fn: impl FnMut(usize, &Tensor, Tensor) -> Result<Tensor, TchError>) -> Result<(), TchError> {
        match self {
            Parameters::Standard(parameters) => {
                tch::no_grad(|| {
                    for (i, param) in parameters.iter_mut().enumerate() {
                        let update = update_fn(i, param, param.f_grad()?)?;
                        let _ = param.f_sub_(&update)?;
                    }
                    Ok(())
                })
            }
            Parameters::Private { parameters, max_grad_norm, noise_multiplier, loss_type } => {
                tch::no_grad(|| {
                    let mut per_param_norms = Vec::with_capacity(parameters.len());
                    for param in parameters.iter() {
                        let per_sample_grad = param.grad();
                        let dims: Vec<i64> = (1..per_sample_grad.dim()).map(|x| x as i64).collect();
                        per_param_norms.push(per_sample_grad.f_norm_scalaropt_dim(2, &dims, false)?);
                    }
                    let per_sample_norms = Tensor::f_stack(&per_param_norms, 1)?
                        .f_norm_scalaropt_dim(2, &[1], false)?;
                    let max_grad_norm = Tensor::of_slice(&[*max_grad_norm as f32]);
                    let per_sample_clip_factor = max_grad_norm.f_div(&per_sample_norms.f_add_scalar(1e-6)?)?.f_clamp(0., 1.)?;
        
                    for (i, param) in parameters.iter_mut().enumerate() {
                        let per_sample_grad = param.grad();
                        let mut update_size = per_sample_grad.size();
                        update_size.remove(0);
                        let grad = Tensor::f_einsum("i,i...", &[&per_sample_clip_factor, &per_sample_grad])?;
                        let mut grad = grad.f_add(&generate_noise_like(&grad, *noise_multiplier)?)?.f_view(&update_size[..])?;
                        if let LossType::Mean(batch_size) = loss_type {
                            let _ = grad.f_div_scalar_(*batch_size)?;
                        }
                        let update = update_fn(i, &param.i(0), grad)?;
                        let _ = param.i(0).f_sub_(&update)?;
                    }
                    Ok(())
                })
            }
        }
    }
}

pub enum LossType {
    Sum,
    Mean(i64),
}

fn generate_noise_like(tensor: &Tensor, std: f64) -> Result<Tensor, TchError> {
    let zeros = Tensor::zeros(&tensor.size(), (Kind::Float, tensor.device()));
    if std == 0. {
        Ok(zeros)
    } else {
        let _ = Tensor::zeros(&[1, 1], (Kind::Float, tensor.device())).f_normal(0., std);
        let mut sum = zeros;
        for _ in 0..4 {
            let _ = sum.f_add_(&Tensor::zeros(&tensor.size(), (Kind::Float, tensor.device())).f_normal(0., std)?);
        }
        let _ = sum.f_div_scalar_(2.);
        Ok(sum)
    }
}

pub fn initialize_statistics(length: usize) -> Vec<Option<Tensor>> {
    let mut v = Vec::with_capacity(length);
    for _ in 0..length {
        v.push(None);
    }
    v
}

pub struct SGD {
    learning_rate: f64,
    weight_decay: f64,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    statistics: Vec<Option<Tensor>>,
    pub parameters: Parameters,
}

impl SGD {
    pub fn new(parameters: Parameters, learning_rate: f64) -> Self {
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
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
    pub fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
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
                    *b = b.f_mul_scalar(self.momentum)?.f_add(&grad.f_mul_scalar(1. - self.dampening)?)?;
                } else {
                    self.statistics[i] = Some(grad.f_detach_copy()?)
                }
                if self.nesterov {
                    // grad = grad + momentum * statistics
                    grad = grad.f_add(&(&self.statistics[i]).as_ref().unwrap().f_mul_scalar(self.momentum)?)?;
                } else {
                    grad = (&self.statistics[i]).as_ref().unwrap().f_detach_copy()?;
                }
            }
            // update = learning_rate * grad
            grad.f_mul_scalar(self.learning_rate)
        })
    }
}

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
