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
        let parameters = PrivateParameters::new(vs, 1.0, 0.1, LossType::Mean(2));
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

pub trait Updatable {
    fn zero_grad(&mut self);
    fn update(&mut self, update_fn: impl FnMut(&Tensor, Tensor) -> Result<Tensor, TchError>) -> Result<(), TchError>;
}

pub struct Parameters(Vec<Tensor>);

impl From<VarStore> for Parameters {
    fn from(vs: VarStore) -> Self {
        Parameters(vs.trainable_variables())
    }
}

impl Parameters {
    pub fn into_inner(mut self) -> Vec<Tensor> {
        self.zero_grad();
        self.0
    }
}

impl Updatable for Parameters {
    fn zero_grad(&mut self) {
        for param in self.0.iter_mut() {
            param.zero_grad();
        }
    }
    fn update(&mut self, mut update_fn: impl FnMut(&Tensor, Tensor) -> Result<Tensor, TchError>) -> Result<(), TchError> {
        tch::no_grad(|| {
            for param in self.0.iter_mut() {
                let update = update_fn(param, param.f_grad()?)?;
                let _ = param.f_sub_(&update)?;
            }
            Ok(())
        })
    }
}

pub enum LossType {
    Sum,
    Mean(i64),
}

pub struct PrivateParameters {
    pub(crate) parameters: Vec<Tensor>,
    max_grad_norm: f64,
    noise_multiplier: f64,
    loss_type: LossType,
}

impl PrivateParameters {
    pub fn new(vs: VarStore, max_grad_norm: f64, noise_multiplier: f64, loss_type: LossType) -> Self {
        PrivateParameters { parameters: vs.trainable_variables(), max_grad_norm, noise_multiplier, loss_type }
    }

    // It is ok to retrieve the weights through
    // this function as we erase the gradients prior
    // to returning. Private data remain private.
    pub fn into_inner(mut self) -> Vec<Tensor> {
        self.zero_grad();
        self.parameters
    }
}

impl Updatable for PrivateParameters {
    fn zero_grad(&mut self) {
        for param in self.parameters.iter_mut() {
            param.zero_grad();
        }
    }
    fn update(&mut self, mut update_fn: impl FnMut(&Tensor, Tensor) -> Result<Tensor, TchError>) -> Result<(), TchError> {
        tch::no_grad(|| {
            let mut per_param_norms = Vec::with_capacity(self.parameters.len());
            for param in self.parameters.iter() {
                let per_sample_grad = param.grad();
                let dims: Vec<i64> = (1..per_sample_grad.dim()).map(|x| x as i64).collect();
                per_param_norms.push(per_sample_grad.f_norm_scalaropt_dim(2, &dims, false)?);
            }
            let per_sample_norms = Tensor::f_stack(&per_param_norms, 1)?
                .f_norm_scalaropt_dim(2, &[1], false)?;
            let max_grad_norm = Tensor::of_slice(&[self.max_grad_norm as f32]);
            let per_sample_clip_factor = max_grad_norm.f_div(&per_sample_norms.f_add_scalar(1e-6)?)?.f_clamp(0., 1.)?;

            for param in self.parameters.iter_mut() {
                let per_sample_grad = param.grad();
                let mut update_size = per_sample_grad.size();
                update_size.remove(0);
                let grad = Tensor::f_einsum("i,i...", &[&per_sample_clip_factor, &per_sample_grad])?;
                let mut grad = grad.f_add(&generate_noise_like(&grad, self.noise_multiplier)?)?.f_view(&update_size[..])?;
                if let LossType::Mean(batch_size) = self.loss_type {
                    let _ = grad.f_div_scalar_(batch_size)?;
                }
                let update = update_fn(&param.i(0), grad)?;
                let _ = param.i(0).f_sub_(&update)?;
            }
            Ok(())
        })
    }
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

pub struct SGD<P> {
    learning_rate: f64,
    weight_decay: f64,
    momentum: f64,
    dampening: f64,
    nesterov: bool,
    pub parameters: P,
    statistics: Option<Tensor>,
}

impl<P> SGD<P> {
    pub fn new(parameters: P, learning_rate: f64) -> Self {
        SGD {
            learning_rate: learning_rate,
            weight_decay: 0.,
            momentum: 0.,
            dampening: 0.,
            nesterov: false,
            parameters,
            statistics: None,
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

impl<P: Updatable> Optimizer for SGD<P> {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        self.parameters.zero_grad();
        Ok(())
    }

    fn step(&mut self) -> Result<(), TchError> {
        self.parameters.update(|x, mut grad| {
            if self.weight_decay != 0. {
                // grad = grad + self.weight_decay * x;
                grad = grad.f_add(&x.f_mul_scalar(self.weight_decay)?)?;
            }
            if self.momentum != 0. {
                if let Some(b) = &mut self.statistics {
                    // b = self.momentum * b + (1 - self.dampening) * grad;
                    *b = b.f_mul_scalar(self.momentum)?.f_add(&grad.f_mul_scalar(1. - self.dampening)?)?;
                } else {
                    self.statistics = Some(grad.f_detach_copy()?)
                }
                if self.nesterov {
                    // grad = grad + self.momentum * self.statistics.unwrap();
                    grad = grad.f_add(&(&self.statistics).as_ref().unwrap().f_mul_scalar(self.momentum)?)?;
                } else {
                    grad = (&self.statistics).as_ref().unwrap().f_detach_copy()?;
                }
            }
            // update := self.learning_rate * grad;
            grad.f_mul_scalar(self.learning_rate)
        })
    }
}
