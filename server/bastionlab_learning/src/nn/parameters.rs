use std::{collections::HashMap, marker::PhantomData};

use super::{module::DpSGDContext, Module};
use crate::data::privacy_guard::{compute_sigma, generate_noise_like, PrivacyBudget};
use std::sync::{Arc, RwLock};
use tch::{nn::VarStore, IndexOp, TchError, Tensor};

/// Securely copies the parameters to avoid leaking gradients.
fn copy_parameters(params: &HashMap<String, Tensor>) -> Result<Vec<Tensor>, TchError> {
    let mut res = Vec::new();
    for (_, p) in params.iter() {
        res.push(p.copy().f_detach_()?);
    }
    Ok(res)
}
/// Securely copies the parameters as [`Vec<u8>`] to avoid leaking gradients.
fn save_to_stream(params: &HashMap<String, Tensor>) -> Result<Vec<u8>, TchError> {
    let mut stream: Vec<u8> = Vec::new();
    let named_tensors = params.iter().collect::<Vec<_>>();
    Tensor::save_multi_to_stream(named_tensors.as_slice(), &mut stream)?;
    Ok(stream.clone())
}

/// Type of batch aggregation used by a loss function
///
/// The `Mean` variant contains the number of samples in a batch.
#[derive(Debug)]
pub enum LossType {
    Sum,
    Mean(i64),
}

/// Contains the trainable parameters of a model to be used by an optimizer
///
/// The standard variant provides standard parameter update, the private variant performs DP-SGD.
/// Note that the private variant requires the model to use expanded weights. In the Python API,
/// layers with expanded weights may be found under `bastionai.psg.nn`. A standard model may also
/// be turned into an expanded one using the `bastionai.psg.expand` function.
#[derive(Debug)]
pub enum Parameters<'a> {
    Standard {
        parameters: HashMap<String, Tensor>,
        dp_sgd_context: Arc<RwLock<Option<DpSGDContext>>>,
        _phantom: PhantomData<&'a mut Module>,
    },
    Private {
        parameters: HashMap<String, Tensor>,
        eps: f32,
        max_grad_norm: f32,
        loss_type: LossType,
        dp_sgd_context: Arc<RwLock<Option<DpSGDContext>>>,
        steps: usize,
        _phantom: PhantomData<&'a mut Module>,
    },
}

impl<'a> Parameters<'a> {
    /// Creates a new standard variant from given `VarStore`.
    pub(crate) fn standard(
        vs: &'a mut VarStore,
        dp_sgd_context: Arc<RwLock<Option<DpSGDContext>>>,
    ) -> Parameters<'a> {
        Parameters::Standard {
            parameters: vs.variables(),
            dp_sgd_context,
            _phantom: PhantomData,
        }
    }

    /// Creates a new private variant from given `VarStore` with given DP parameters.
    ///
    /// `max_grad_norm` controls gradient clipping.
    /// `noise_multiplier` controls the level of DP noise to apply.
    /// `loss_type` tells the DP-SGD algorithm which type of aggregation is used by the training loss: either sum or mean.
    pub(crate) fn private(
        vs: &'a mut VarStore,
        eps: f32,
        max_grad_norm: f32,
        loss_type: LossType,
        dp_sgd_context: Arc<RwLock<Option<DpSGDContext>>>,
    ) -> Parameters<'a> {
        Parameters::Private {
            parameters: vs.variables(),
            eps,
            max_grad_norm,
            loss_type,
            dp_sgd_context,
            steps: 0,
            _phantom: PhantomData,
        }
    }

    /// Returns contained parameters.
    ///
    /// This method is useful to inspect the weights during or after training.
    /// Note that for privacy reasons, this method actually returns a copy of
    /// the parameters that do not contain the accumulated gradients because
    /// the gradients contain non DP protected information about the samples.
    pub fn into_inner(&self) -> Result<Vec<Tensor>, TchError> {
        // self.zero_grad();
        match self {
            Parameters::Standard { parameters, .. } => copy_parameters(parameters),
            Parameters::Private { parameters, .. } => copy_parameters(parameters),
        }
    }
    /// Returns contained parameters as [`Vec<u8>`].
    ///
    /// This method is useful to inspect the weights during or after training.
    /// Note that for privacy reasons, this method actually returns a copy of
    /// the parameters that do not contain the accumulated gradients because
    /// the gradients contain non DP protected information about the samples.
    pub fn into_bytes(&self) -> Result<Vec<u8>, TchError> {
        match self {
            Parameters::Standard { parameters, .. } => save_to_stream(parameters),
            Parameters::Private { parameters, .. } => save_to_stream(parameters),
        }
    }

    /// Returns the number of contained parameters.
    pub fn len(&self) -> usize {
        match self {
            Parameters::Standard { parameters, .. } => parameters.len(),
            Parameters::Private { parameters, .. } => parameters.len(),
        }
    }

    /// Sets all accumulated gradients to zero.
    pub fn zero_grad(&mut self) {
        match self {
            Parameters::Standard { parameters, .. } => {
                for (_, param) in parameters.iter_mut() {
                    param.zero_grad();
                }
            }
            Parameters::Private { parameters, .. } => {
                for (_, param) in parameters.iter_mut() {
                    param.zero_grad();
                }
            }
        }
    }
    /// Overrides model parameters with saved update.
    pub fn override_parameters(&mut self, params: Vec<(String, Tensor)>) -> Result<(), TchError> {
        match self {
            Parameters::Standard { parameters, .. } => {
                for (name, param) in params.iter() {
                    let param0 = match parameters.get_mut(name) {
                        Some(v) => v,
                        None => {
                            return Err(TchError::Convert(
                                "Unable to fetch module parameters".to_string(),
                            ));
                        }
                    };
                    tch::no_grad(|| -> Result<(), TchError> {
                        let _ = param0.f_zero_()?;
                        let _ = param0.f_add_(&param)?;
                        Ok(())
                    })?;
                }
            }
            Parameters::Private { parameters, .. } => {
                for (name, param) in params.iter() {
                    let param0 = match parameters.get_mut(name) {
                        Some(v) => v,
                        None => {
                            return Err(TchError::Convert(
                                "Unable to fetch module parameters".to_string(),
                            ));
                        }
                    };
                    tch::no_grad(|| -> Result<(), TchError> {
                        let _ = param0.i(0).f_zero_()?;
                        let _ = param0.i(0).f_add_(&param.i(0))?;
                        Ok(())
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Iterates over the contained parameters and updates them using given update function.
    ///
    /// When called on a private variant, DP-SGD is applied.
    pub fn update(
        &mut self,
        mut update_fn: impl FnMut(&String, &Tensor, Tensor) -> Result<Tensor, TchError>,
    ) -> Result<(), TchError> {
        match self {
            Parameters::Standard {
                parameters,
                dp_sgd_context,
                ..
            } => tch::no_grad(|| {
                if !dp_sgd_context
                    .read()
                    .expect("Poisoned lock")
                    .as_ref()
                    .unwrap()
                    .empty_guard()
                    .within_bounds(PrivacyBudget::NotPrivate)
                {
                    return Err(TchError::Kind(String::from("Privacy limit violation.")));
                }
                for (name, param) in parameters.iter_mut() {
                    let update = update_fn(name, param, param.f_grad()?)?;
                    let _ = param.f_sub_(&update)?;
                    dp_sgd_context
                        .write()
                        .expect("Poisoned lock")
                        .as_mut()
                        .unwrap()
                        .empty_guard()
                        .empty()
                        .get_private(PrivacyBudget::NotPrivate)?;
                }
                Ok(())
            }),
            Parameters::Private {
                parameters,
                eps,
                max_grad_norm,
                loss_type,
                dp_sgd_context,
                steps,
                ..
            } => tch::no_grad(|| {
                let t = *steps as f32;
                *steps += 1;
                let delta = dp_sgd_context.read().unwrap().as_ref().unwrap().delta();
                let batch_sampling_rate = dp_sgd_context
                    .read()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .batch_sampling_rate();
                let budget_update = *eps * batch_sampling_rate * ((t + 1.0).sqrt() - t.sqrt());
                let sigma = compute_sigma(*eps, delta, *max_grad_norm) as f64;

                if !dp_sgd_context
                    .read()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .empty_guard()
                    .within_bounds(PrivacyBudget::Private(budget_update))
                {
                    return Err(TchError::Kind(String::from("Privacy limit violation.")));
                }

                let mut per_param_norms = Vec::with_capacity(parameters.len());
                for (_, param) in parameters.iter() {
                    let per_sample_grad = param.grad();
                    let dims: Vec<i64> = (1..per_sample_grad.dim()).map(|x| x as i64).collect();
                    per_param_norms.push(per_sample_grad.f_norm_scalaropt_dim(2, &dims, false)?);
                }
                let per_sample_norms =
                    Tensor::f_stack(&per_param_norms, 1).map_err(|e| TchError::Shape(format!("Failed to stack per-sample gradients, are you using a model with expanded weights? Initial error: {}", e)))?.f_norm_scalaropt_dim(2, &[1], false)?;
                let max_grad_norm_t = Tensor::of_slice(&[*max_grad_norm as f32])
                    .f_to_device(per_sample_norms.device())?;
                let per_sample_clip_factor = max_grad_norm_t
                    .f_div(&per_sample_norms.f_add_scalar(1e-6)?)?
                    .f_clamp(0., 1.)?;

                for (i, (name, param)) in parameters.iter_mut().enumerate() {
                    let per_sample_grad = param.grad();
                    let mut update_size = per_sample_grad.size();
                    update_size.remove(0);
                    let grad = Tensor::f_einsum(
                        "i,i...",
                        &[&per_sample_clip_factor, &per_sample_grad],
                        None,
                    )?;
                    let mut grad = grad
                        .f_add(&generate_noise_like(&grad, sigma)?)?
                        .f_view(&update_size[..])?;
                    if let LossType::Mean(batch_size) = loss_type {
                        let _ = grad.f_div_scalar_(*batch_size)?;
                    }
                    let update = update_fn(name, &param.i(0), grad)?;
                    let _ = param.i(0).f_sub_(&update)?;
                    if i == 0 {
                        dp_sgd_context
                            .write()
                            .unwrap()
                            .as_mut()
                            .unwrap()
                            .empty_guard()
                            .empty()
                            .get_private(PrivacyBudget::Private(budget_update))?;
                    }
                }
                Ok(())
            }),
        }
    }
}
