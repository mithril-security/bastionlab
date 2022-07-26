use tch::{TchError, TrainableCModule, COptimizer, Tensor, IValue, Kind, Device};

#[cfg(test)]
mod tests {
    use tch::{TrainableCModule, Device, Tensor, Kind, TchError};
    use tch::nn::VarStore;

    use crate::{Parameters, PrivateParameters, LossType, SGD, Optimizer};

    fn l2_loss(output: &Tensor, target: &Tensor) -> Result<Tensor, TchError> {
        output.f_sub(&target)?.f_norm_scalaropt_dim(2, &[1], false)?.f_mean(Kind::Float)
    }

    #[test]
    fn basic_sgd() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/lreg.pt", vs.root()).unwrap();
        let parameters = Parameters::try_new(&model).unwrap();
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
        let w = &vs.trainable_variables()[0];
        assert!((w - Tensor::of_slice::<f32>(&[2.])).abs().double_value(&[]) < 0.1);
    }

    #[test]
    fn private_sgd() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/private_lreg.pt", vs.root()).unwrap();
        let parameters = PrivateParameters::try_new(&model, 1000.0, 0.0, LossType::Mean(2)).unwrap();
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
                let parameters2 = PrivateParameters::try_new(&model, 1000.0, 0.0, LossType::Mean(2)).unwrap();
                parameters2.parameters[0].1.print();
                optimizer.step().unwrap();
            }
        }
        let w = &vs.trainable_variables()[0];
        assert!(l2_loss(w, &Tensor::of_slice::<f32>(&[2.])).unwrap().double_value(&[]) < 0.1);
    }

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
    fn update(&mut self, update_fn: impl FnMut(&mut Tensor, Tensor) -> Result<(), TchError>) -> Result<(), TchError>;
}

pub struct Parameters(Vec<Tensor>);

impl Parameters {
    pub fn try_new(module: &TrainableCModule) -> Result<Self, TchError> {
        let err = Err(TchError::FileFormat(String::from("Invalid data, expected module to have a `trainable_parameters` function returning the dict of named trainable parameters.")));
        Ok(Parameters(match module.method_is::<IValue>("trainable_parameters", &[])? {
            IValue::GenericDict(v) => {
                let mut res = Vec::with_capacity(v.len());
                for (_, parameter) in v {
                    match parameter {
                        IValue::Tensor(t) => res.push(t),
                        _ => return err,
                    }
                }
                res
            }
            _ => return err,
        }))
    }
}

impl Updatable for Parameters {
    fn update(&mut self, mut update_fn: impl FnMut(&mut Tensor, Tensor) -> Result<(), TchError>) -> Result<(), TchError> {
        tch::no_grad(|| {
            for param in self.0.iter_mut() {
                update_fn(param, param.f_grad()?)?;
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
    pub(crate) parameters: Vec<(Tensor, Tensor)>,
    max_grad_norm: f64,
    noise_multiplier: f64,
    loss_type: LossType,
}

impl PrivateParameters {
    pub fn try_new(module: &TrainableCModule, max_grad_norm: f64, noise_multiplier: f64, loss_type: LossType) -> Result<Self, TchError> {
        let err = Err(TchError::FileFormat(String::from("Invalid data, expected module to have a `grad_sample_parameters` function returning the list of pairs of parameters and per-sample gradients.")));
        let parameters = match module.method_is::<IValue>("grad_sample_parameters", &[])? {
            IValue::GenericList(v) => {
                let mut res = Vec::with_capacity(v.len());
                for x in v {
                    match x {
                        IValue::Tuple(mut t) => {
                            match t.remove(0) {
                                IValue::Tensor(param) => {
                                    match t.remove(0) {
                                        IValue::Tensor(grad) => res.push((param, grad)),
                                        _ => return err,
                                    }
                                },
                                _ => return err,
                            }
                        }
                        _ => return err,
                    }
                }
                res
            }
            _ => return err,
        };
        Ok(PrivateParameters { parameters, max_grad_norm, noise_multiplier, loss_type })
    }
}

impl Updatable for PrivateParameters {
    fn update(&mut self, mut update_fn: impl FnMut(&mut Tensor, Tensor) -> Result<(), TchError>) -> Result<(), TchError> {
        tch::no_grad(|| {
            let mut per_param_norms = Vec::with_capacity(self.parameters.len());
            for (_, grad_sample) in self.parameters.iter() {
                let dims: Vec<i64> = (1..grad_sample.dim()).map(|x| x as i64).collect();
                per_param_norms.push(grad_sample.f_norm_scalaropt_dim(2, &dims, false)?);
            }
            let per_sample_norms = Tensor::f_stack(&per_param_norms, 1)?
                .f_norm_scalaropt_dim(2, &[1], false)?;
            let max_grad_norm = Tensor::of_slice(&[self.max_grad_norm as f32]);
            let per_sample_clip_factor = max_grad_norm.f_div(&per_sample_norms.f_add_scalar(1e-6)?)?.f_clamp(0., 1.)?;

            for (param, grad_sample) in self.parameters.iter_mut() {
                grad_sample.print();
                let grad = Tensor::f_einsum("i,i...", &[&per_sample_clip_factor, &grad_sample])?;
                // grad.print();
                let mut grad = grad.f_add(&generate_noise_like(&grad, self.noise_multiplier)?)?.f_view_as(&param)?;
                // grad.print();
                if let LossType::Mean(batch_size) = self.loss_type {
                    let _ = grad.f_div_scalar_(batch_size)?;
                }
                // grad.print();
                update_fn(param, grad)?;
                // param.print();
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
    parameters: P,
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
        self.parameters.update(|x, _| {
            x.zero_grad();
            Ok(())
        })
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
            // x = x - self.learning_rate * grad;
            let _ = x.f_sub_(&grad.f_mul_scalar(self.learning_rate)?)?.f_set_requires_grad(true)?;
            Ok(())
        })
    }
}
