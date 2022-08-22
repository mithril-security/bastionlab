use tch::{Tensor, nn::VarStore, TchError, Kind, IndexOp};

// Generates a tensor having the same size as `tensor` that contains gaussian noise
// with mean 0 and standard deviation `std`.
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

/// Type of batch aggregation used by a loss function
/// 
/// The `Mean` variant contains the number of samples in a batch.
pub enum LossType {
    Sum,
    Mean(i64),
}

/// Contains the trainable parameters of a model to be used by an optimizer
/// 
/// The standard variant provides standard parameter update, the private variant performs DP-SGD.
/// Note that the private variant requires the model to use expanded weights. In the Python API,
/// layers with expanded weights may be found under `bastionai.psg.nn`.
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
    /// Creates a new standard variant from given `VarStore`.
    pub fn standard(vs: &VarStore) -> Self {
        Parameters::Standard(vs.trainable_variables())
    }

    /// Creates a new private variant from given `VarStore` with given DP parameters.
    /// 
    /// `max_grad_norm` controls gradient clipping.
    /// `noise_multiplier` controls the level of DP noise to apply.
    /// `loss_type` tells the DP-SGD algorithm which type of aggregation is used by the training loss: either sum or mean.
    pub fn private(vs: &VarStore, max_grad_norm: f64, noise_multiplier: f64, loss_type: LossType) -> Self {
        Parameters::Private { parameters: vs.trainable_variables(), max_grad_norm, noise_multiplier, loss_type }
    }

    /// Returns contained parameters.
    /// 
    /// This method is useful to inspect the weights during or after training.
    /// Note that for privacy reasons, a call to this method erases the accumulated gradients
    /// that contain non DP protected information about the samples.
    pub fn into_inner(mut self) -> Vec<Tensor> {
        self.zero_grad();
        match self {
            Parameters::Standard(parameters) => parameters,
            Parameters::Private { parameters, .. } => parameters,
        }
    }

    /// Returns the number of contained parameters.
    pub fn len(&self) -> usize {
        match self {
            Parameters::Standard(parameters) => parameters.len(),
            Parameters::Private { parameters, .. } => parameters.len(),
        }
    }

    /// Sets all accumulated gradients to zero.
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

    /// Iterates over the contained parameters and updates them using given update function.
    /// 
    /// When called on a private variant, DP-SGD is applied.
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
                        per_param_norms.push(per_sample_grad.f_norm_scalaropt_dim(2, &dims, false).unwrap());
                    }
                    let per_sample_norms = Tensor::f_stack(&per_param_norms, 1).unwrap()
                        .f_norm_scalaropt_dim(2, &[1], false).unwrap();
                    let max_grad_norm = Tensor::of_slice(&[*max_grad_norm as f32]);
                    let per_sample_clip_factor = max_grad_norm.f_div(&per_sample_norms.f_add_scalar(1e-6).unwrap()).unwrap().f_clamp(0., 1.).unwrap();
        
                    for (i, param) in parameters.iter_mut().enumerate() {
                        let per_sample_grad = param.grad();
                        let mut update_size = per_sample_grad.size();
                        update_size.remove(0);
                        let grad = Tensor::f_einsum("i,i...", &[&per_sample_clip_factor, &per_sample_grad]).unwrap();
                        let mut grad = grad.f_add(&generate_noise_like(&grad, *noise_multiplier).unwrap()).unwrap().f_view(&update_size[..]).unwrap();
                        if let LossType::Mean(batch_size) = loss_type {
                            let _ = grad.f_div_scalar_(*batch_size).unwrap();
                        }
                        let update = update_fn(i, &param.i(0), grad).unwrap();
                        let _ = param.i(0).f_sub_(&update).unwrap();
                    }
                    Ok(())
                })
            }
        }
    }
}
