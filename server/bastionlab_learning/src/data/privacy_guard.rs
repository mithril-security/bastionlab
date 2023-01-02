use crate::nn::Forward;
use std::borrow::Borrow;
use std::ops::Add;
use std::sync::{Arc, RwLock};
use tch::{Device, Kind, Reduction, Shape, TchError, Tensor};

/// Generates a tensor having the same size as `tensor` that contains gaussian noise
/// with mean 0 and standard deviation `std`.
pub(crate) fn generate_noise_like(tensor: &Tensor, std: f64) -> Result<Tensor, TchError> {
    let zeros = Tensor::zeros(&tensor.size(), (Kind::Float, tensor.device()));
    if std == 0. {
        Ok(zeros)
    } else {
        let mut sum = zeros;
        for _ in 0..4 {
            let _ = sum.f_add_(
                &Tensor::zeros(&tensor.size(), (Kind::Float, tensor.device()))
                    .f_normal_functional(0., std)?,
            );
        }
        let _ = sum.f_div_scalar_(2.);
        Ok(sum)
    }
}

/// Returns the std deviation of the gaussian noise that must be added to attain
/// (`eps`-`delta`)-DP guarantees with a mechanism having the given `l2_sensibility`.
pub(crate) fn compute_sigma(eps: f32, delta: f32, l2_sensibility: f32) -> f32 {
    l2_sensibility * (2.0 * (1.25 / delta).ln()).sqrt() / (eps + 1e-8)
}

/// Checks whether sets `a` and `b` are disjoint.
fn independence_check<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    let mut res = true;
    for x in b.iter() {
        res = res && !a.contains(x);
    }
    res
}

/// Encodes the sensibility of a function evaluation w.r.t. its input expressed
/// either in `LInfinity` or `L2` norm. The `Unknown` variant is used
/// when the sensibility cannot be automatically infered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sensibility {
    Unknown,
    L2(f32),
    LInfinity(f32),
}

/// Encodes the interdependence of a tensor's lines.
///
/// A tensor is considered `Independent` if each line
/// is computed independently of the others using a
/// single sample from the dataset (this is usually
/// the case when batching), and `Dependent` otherwise.
///
/// The `Idenpendent` variant contains a list of batches
/// involved in the computation. This is useful to mark
/// the result of a bianry operation with two independent
/// tensors as independent if the two tensors were computed
/// on two disjoint sets of batches. For this purpose, this
/// type implepents Add.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchDependence {
    Dependent,
    Independent(Vec<usize>),
}

impl Add for BatchDependence {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BatchDependence::Dependent, _) => BatchDependence::Dependent,
            (_, BatchDependence::Dependent) => BatchDependence::Dependent,
            (BatchDependence::Independent(mut a), BatchDependence::Independent(mut b)) => {
                if independence_check(&a, &b) {
                    a.append(&mut b);
                    BatchDependence::Independent(a)
                } else {
                    BatchDependence::Dependent
                }
            }
        }
    }
}

/// Represents a privacy budget (epsilon) that is possibly infinite (`NotPrivate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrivacyBudget {
    NotPrivate,
    Private(f32),
}

/// Contains all the necessary data for [`Dataset`] to track its usage.
///
/// This struct is placed in an [`Arc`] and shared with all PrivacyGuards
/// that contain data from the dataset so that the guards can increase the
/// expended budget when turned into readable values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrivacyContext {
    expended: PrivacyBudget,
    limit: PrivacyBudget,
    delta: f32,
    nb_samples: usize,
}

impl PrivacyContext {
    pub fn new(limit: PrivacyBudget, nb_samples: usize) -> Self {
        PrivacyContext {
            expended: PrivacyBudget::Private(0.0),
            limit,
            delta: 1.0 / (10.0 * nb_samples as f32),
            nb_samples,
        }
    }

    pub fn within_bounds(&self, budget: PrivacyBudget) -> bool {
        match &self.limit {
            PrivacyBudget::NotPrivate => true,
            PrivacyBudget::Private(eps_limit) => match &self.expended {
                PrivacyBudget::NotPrivate => false,
                PrivacyBudget::Private(eps_expended) => match budget {
                    PrivacyBudget::NotPrivate => false,
                    PrivacyBudget::Private(eps) => eps + eps_expended < *eps_limit,
                },
            },
        }
    }

    pub fn delta(&self) -> f32 {
        self.delta
    }

    pub fn nb_samples(&self) -> usize {
        self.nb_samples
    }

    fn update_budget(&mut self, budget: PrivacyBudget) {
        match (&mut self.expended, budget) {
            (PrivacyBudget::NotPrivate, _) => (),
            (PrivacyBudget::Private(_), PrivacyBudget::NotPrivate) => {
                self.expended = PrivacyBudget::NotPrivate
            }
            (PrivacyBudget::Private(eps_expended), PrivacyBudget::Private(eps)) => {
                *eps_expended += eps;
            }
        }
    }
}

/// Encapsulate a private result.
///
/// It is impossible to read the value of the result without
/// using either the `get_private` or `get_not_private` methods.
///
/// `get_private` returns a noised value of the contained result according
/// to the allocated privacy budget if the dataset's privacy limit allows it.
/// It also updates the dataset's expended budget accordingly.
///
/// `get_non_private` returns the unnoised value if the dataset allows it
/// (no privacy limit is set) and sets the dataset's expended budget to `NotPrivate`
/// (infinite).
#[derive(Debug, Clone)]
pub struct PrivacyGuard<T> {
    value: T,
    sensibility: Sensibility,
    batch_dependence: BatchDependence,
    context: Arc<RwLock<PrivacyContext>>,
}

impl<T> PrivacyGuard<T> {
    pub fn new(
        value: T,
        batch_dependence: BatchDependence,
        context: Arc<RwLock<PrivacyContext>>,
    ) -> Self {
        PrivacyGuard {
            value,
            sensibility: Sensibility::Unknown,
            batch_dependence,
            context,
        }
    }

    /// Returns a noised value of the contained result according
    /// to the allocated privacy budget if the dataset's privacy limit allows it.
    /// It also updates the dataset's expended budget accordingly.
    pub fn get_non_private(self) -> T {
        self.context.write().unwrap().expended = PrivacyBudget::NotPrivate;
        self.value
    }

    /// Applies an immutable function to the conatained shared context.
    ///
    /// Useful to read dataset level information such as delta, privacy limit, etc.
    pub fn map_context<U>(&self, f: impl Fn(&PrivacyContext) -> U) -> U {
        f(&self.context.read().unwrap())
    }

    /// Returns an empty guard that shares the same context.
    ///
    /// Useful to enforce a privacy contraint without actually retrieving any value.
    pub fn empty(&self) -> PrivacyGuard<()> {
        PrivacyGuard {
            value: (),
            sensibility: Sensibility::L2(0.0),
            batch_dependence: BatchDependence::Independent(vec![]),
            context: Arc::clone(&self.context),
        }
    }

    /// Checks whether the value can privately be retrieved with given budget.
    pub fn within_bounds(&self, budget: PrivacyBudget) -> bool {
        self.context.read().unwrap().within_bounds(budget)
    }
}

impl PrivacyGuard<f32> {
    pub fn into_double(self) -> PrivacyGuard<f64> {
        PrivacyGuard {
            value: self.value as f64,
            sensibility: self.sensibility,
            batch_dependence: self.batch_dependence,
            context: self.context,
        }
    }
}

impl PrivacyGuard<f64> {
    pub fn into_float(self) -> PrivacyGuard<f32> {
        PrivacyGuard {
            value: self.value as f32,
            sensibility: self.sensibility,
            batch_dependence: self.batch_dependence,
            context: self.context,
        }
    }
}

macro_rules! defer_f_fns_to_inner {
    ($($fn:ident(
        $self:ident: $self_type:ty$(,
        $arg0:ident: $type0:ty$(,
        $arg1:ident: $type1:ty$(,
        $arg2:ident: $type2:ty$(,
        $arg3:ident: $type3:ty$(,
        $arg4:ident: $type4:ty)?)?)?)?)?
    ) -> $out_type:ty where sensibility = $sensibility:expr, batch_dependence = $batch_dependence:expr)*) => {$(
        pub fn $fn(
            $self: $self_type$(,
            $arg0: $type0$(,
            $arg1: $type1$(,
            $arg2: $type2$(,
            $arg3: $type3$(,
            $arg4: $type4)?)?)?)?)?
        ) -> $out_type {
            Ok(PrivacyGuard {
                sensibility: $sensibility,
                batch_dependence: $batch_dependence,
                value: $self.value.$fn($($arg0$(, $arg1$(, $arg2$(, $arg3$(, $arg4)?)?)?)?)?)?,
                context: Arc::clone(&$self.context),
            })
        }
    )*};
}

macro_rules! defer_binary_f_fns_to_inner {
    ($($fn:ident(
        $self:ident: $self_type:ty,
        $arg0:ident: $type0:ty$(,
        $arg1:ident: $type1:ty$(,
        $arg2:ident: $type2:ty$(,
        $arg3:ident: $type3:ty$(,
        $arg4:ident: $type4:ty)?)?)?)?
    ) -> $out_type:ty where sensibility = $sensibility:expr, batch_dependence = $batch_dependence:expr)*) => {$(
        pub fn $fn(
            $self: $self_type,
            $arg0: $type0$(,
            $arg1: $type1$(,
            $arg2: $type2$(,
            $arg3: $type3$(,
            $arg4: $type4)?)?)?)?
        ) -> $out_type {
            if Arc::as_ptr(&$self.context) != Arc::as_ptr(&$arg0.context) {
                return Err(TchError::Kind(String::from(
                    "Inputs must share the same privacy context.",
                )));
            }
            Ok(PrivacyGuard {
                sensibility: $sensibility,
                batch_dependence: $batch_dependence,
                value: $self.value.$fn(&$arg0.value$(, $arg1$(, $arg2$(, $arg3$(, $arg4)?)?)?)?)?,
                context: Arc::clone(&$self.context),
            })
        }
    )*};
}

macro_rules! defer_loss_fn_to_inner_with_clipping {
    ($($fn:ident(
        $self:ident: $self_type:ty,
        $target:ident: $type_target:ty,
        $clipping:ident: $type_clipping:ty$(,
        $arg1:ident: $type1:ty$(,
        $arg2:ident: $type2:ty$(,
        $arg3:ident: $type3:ty$(,
        $arg4:ident: $type4:ty)?)?)?)?
    ) -> $out_type:ty)*) => {$(
        pub fn $fn(
            $self: $self_type,
            $target: $type_target,
            $clipping: $type_clipping$(,
            $arg1: $type1$(,
            $arg2: $type2$(,
            $arg3: $type3$(,
            $arg4: $type4)?)?)?)?
        ) -> $out_type {
            if Arc::as_ptr(&$self.context) != Arc::as_ptr(&$target.context) {
                return Err(TchError::Kind(String::from(
                    "Inputs must share the same privacy context.",
                )));
            }

            let unreduced = $self.value.$fn(&$target.value$(, $arg1$(, $arg2$(, $arg3$(, $arg4)?)?)?)?)?;
            let clipped = unreduced.f_clamp($clipping.0, $clipping.1)?;
            let max_norm = $clipping.0.abs().max($clipping.1.abs()) as f32;

            let (batch_dependence, sensibility) = match (&$self.batch_dependence, &$target.batch_dependence)
            {
                (BatchDependence::Independent(a), BatchDependence::Independent(b)) if a == b => (
                    BatchDependence::Independent(a.clone()),
                    Sensibility::LInfinity(max_norm),
                ),
                _ => (
                    BatchDependence::Dependent,
                    Sensibility::LInfinity($self.batch_size()? as f32 * max_norm),
                ),
            };

            let non_clipped = PrivacyGuard {
                value: unreduced.f_sum_dim_intlist(Some(&[0i64] as &[_]), false, Kind::Float)?,
                sensibility: Sensibility::Unknown,
                batch_dependence: batch_dependence.clone(),
                context: Arc::clone(&$self.context),
            };

            let clipped = PrivacyGuard {
                value: clipped.f_sum_dim_intlist(Some(&[0i64] as &[_]), false, Kind::Float)?,
                sensibility,
                batch_dependence,
                context: Arc::clone(&$self.context),
            };
            Ok((non_clipped, clipped))
        }
    )*};
}

impl PrivacyGuard<Tensor> {
    pub fn get_private_with_std(self, budget: PrivacyBudget) -> Result<(Tensor, f32), TchError> {
        match budget {
            PrivacyBudget::NotPrivate => Ok((self.get_non_private(), 0.0)),
            PrivacyBudget::Private(eps) => {
                let l2_sensibility = match self.sensibility {
                    Sensibility::Unknown => {
                        return Err(TchError::Kind(String::from(
                            "Unknown sensibility. Consider clipping prior to noising.",
                        )))
                    }
                    Sensibility::LInfinity(s) => (self.value.numel() as f32).sqrt() * s,
                    Sensibility::L2(s) => s,
                };
                let mut context = self.context.write().unwrap();
                if !context.within_bounds(budget) {
                    return Err(TchError::Kind(String::from("Privacy limit violation.")));
                }
                let sigma = compute_sigma(eps, context.delta, l2_sensibility);
                let res = self
                    .value
                    .f_add(&generate_noise_like(&self.value, sigma as f64)?)?;
                context.update_budget(budget);
                Ok((res, sigma))
            }
        }
    }

    pub fn get_private(self, budget: PrivacyBudget) -> Result<Tensor, TchError> {
        Ok(self.get_private_with_std(budget)?.0)
    }

    pub(crate) fn apply_forward<'a>(
        f: Forward<'a>,
        inputs: Vec<PrivacyGuard<Tensor>>,
    ) -> Result<PrivacyGuard<Tensor>, TchError> {
        let (inner_inputs, batch_dependence, context) = if inputs.len() > 0 {
            let ptr = Arc::as_ptr(&inputs[0].context);
            if inputs
                .iter()
                .skip(1)
                .fold(true, |acc, input| Arc::as_ptr(&input.context) == ptr && acc)
            {
                let batch_dependence = inputs[0].batch_dependence.clone();
                let context = Arc::clone(&inputs[0].context);
                if inputs.iter().skip(1).fold(true, |acc, input| {
                    input.batch_dependence == batch_dependence && acc
                }) {
                    (
                        inputs
                            .into_iter()
                            .map(|input| input.value)
                            .collect::<Vec<_>>(),
                        batch_dependence,
                        context,
                    )
                } else {
                    return Err(TchError::Kind(String::from(
                        "Inputs must come from the same batch.",
                    )));
                }
            } else {
                return Err(TchError::Kind(String::from(
                    "Inputs must share the same privacy context.",
                )));
            }
        } else {
            return Err(TchError::Kind(String::from(
                "At least one input must be provided to the model.",
            )));
        };

        Ok(PrivacyGuard {
            value: f.forward_inner(&inner_inputs)?,
            sensibility: Sensibility::Unknown,
            batch_dependence,
            context,
        })
    }

    pub fn batch_size(&self) -> Result<i64, TchError> {
        let size = self.value.size();
        if size.len() > 0 {
            Ok(self.value.size()[0])
        } else {
            Err(TchError::Kind(String::from(
                "Tensor has no dimmensions, cannot infer batch size.",
            )))
        }
    }

    pub fn expand_batch_dim(&self, n: i64) -> Result<Self, TchError> {
        let mut repeats = vec![1; self.value.dim()];
        repeats[0] = n;

        Ok(PrivacyGuard {
            value: self.value.f_repeat(&repeats)?,
            sensibility: Sensibility::Unknown,
            batch_dependence: BatchDependence::Dependent,
            context: Arc::clone(&self.context),
        })
    }

    pub fn backward(&self) {
        self.value.backward();
    }

    pub fn f_clone(&self) -> Result<Self, TchError> {
        Ok(PrivacyGuard {
            value: self.value.copy().f_detach()?,
            sensibility: self.sensibility,
            batch_dependence: self.batch_dependence.clone(),
            context: Arc::clone(&self.context),
        })
    }

    defer_loss_fn_to_inner_with_clipping! {
        f_mse_loss(self: &Self, target: &Self, clipping: (f64, f64), reduction: Reduction) -> Result<(Self, Self), TchError>
        f_cross_entropy_loss(
            self: &Self,
            target: &Self,
            clipping: (f64, f64),
            weight: Option<impl Borrow<Tensor>>,
            reduction: Reduction,
            ignore_index: i64,
            label_smoothing: f64
        ) -> Result<(Self, Self), TchError>
    }

    defer_f_fns_to_inner! {
        f_to(self: &Self, device: Device) -> Result<Self, TchError> where sensibility = self.sensibility, batch_dependence = self.batch_dependence.clone()
        f_view(self: &Self, s: impl Shape) -> Result<Self, TchError> where sensibility = Sensibility::Unknown, batch_dependence = self.batch_dependence.clone()
        f_abs(self: &Self) -> Result<Self, TchError> where sensibility = self.sensibility, batch_dependence = self.batch_dependence.clone()
        f_double_value(self: &Self, idx: &[i64]) -> Result<PrivacyGuard<f64>, TchError> where sensibility = self.sensibility, batch_dependence = self.batch_dependence.clone()
        f_clamp(self: &Self, min: f64, max: f64) -> Result<Self, TchError> where sensibility = match self.sensibility {
            Sensibility::LInfinity(s) => Sensibility::LInfinity(s.min(max as f32 - min as f32)),
            _ => Sensibility::LInfinity((max - min) as f32),
        }, batch_dependence = self.batch_dependence.clone()
        f_sum(self: &Self, dtype: Kind) -> Result<Self, TchError> where sensibility = match self.sensibility {
            Sensibility::Unknown => Sensibility::Unknown,
            Sensibility::LInfinity(s) => Sensibility::LInfinity(match self.batch_dependence {
                BatchDependence::Dependent => self.value.numel() as f32 * s,
                BatchDependence::Independent(_) => s,
            }),
            Sensibility::L2(_) => Sensibility::Unknown,
        }, batch_dependence = self.batch_dependence.clone()
        f_argmax(self: &Self, dim: i64, keepdim: bool) -> Result<Self, TchError> where sensibility = Sensibility::LInfinity(self.value.size()[if dim > 0 { dim as usize } else { self.value.dim() - 1 }] as f32), batch_dependence = self.batch_dependence.clone()
        f_add_scalar(self: &Self, other: impl Into<tch::Scalar>) -> Result<Self, TchError> where sensibility = self.sensibility, batch_dependence = self.batch_dependence.clone()
        f_sub_scalar(self: &Self, other: impl Into<tch::Scalar>) -> Result<Self, TchError> where sensibility = self.sensibility, batch_dependence = self.batch_dependence.clone()
        f_mul_scalar(self: &Self, other: f64) -> Result<Self, TchError> where sensibility = match self.sensibility {
            Sensibility::Unknown => Sensibility::Unknown,
            Sensibility::LInfinity(s) => Sensibility::LInfinity(other.abs() as f32 * s),
            Sensibility::L2(s) => Sensibility::L2(other.abs() as f32 * s),
        }, batch_dependence = self.batch_dependence.clone()
    }

    defer_binary_f_fns_to_inner! {
        f_add(self: &Self, other: &Self) -> Result<Self, TchError> where sensibility = {
            let compose = |a: f32, b: f32| if let BatchDependence::Independent(_) = self.batch_dependence.clone() + other.batch_dependence.clone() {
                a.max(b)
            } else {
                a + b
            };

            match (self.sensibility, other.sensibility) {
                (Sensibility::L2(a), Sensibility::L2(b)) => Sensibility::L2((compose)(a, b)),
                (Sensibility::L2(a), Sensibility::LInfinity(b)) => Sensibility::L2((compose)(a, (other.value.numel() as f32).sqrt() * b)),
                (Sensibility::LInfinity(a), Sensibility::L2(b)) => Sensibility::L2((compose)((self.value.numel() as f32).sqrt() * a, b)),
                (Sensibility::LInfinity(a), Sensibility::LInfinity(b)) => Sensibility::LInfinity((compose)((self.value.numel() as f32).sqrt() * a, (other.value.numel() as f32).sqrt() * b)),
                _ => Sensibility::Unknown,
            }
        }, batch_dependence = self.batch_dependence.clone() + other.batch_dependence.clone()
        f_sub(self: &Self, other: &Self) -> Result<Self, TchError> where sensibility = {
            let compose = |a: f32, b: f32| if let BatchDependence::Independent(_) = self.batch_dependence.clone() + other.batch_dependence.clone() {
                a.max(b)
            } else {
                a + b
            };

            match (self.sensibility, other.sensibility) {
                (Sensibility::L2(a), Sensibility::L2(b)) => Sensibility::L2((compose)(a, b)),
                (Sensibility::L2(a), Sensibility::LInfinity(b)) => Sensibility::L2((compose)(a, (other.value.numel() as f32).sqrt() * b)),
                (Sensibility::LInfinity(a), Sensibility::L2(b)) => Sensibility::L2((compose)((self.value.numel() as f32).sqrt() * a, b)),
                (Sensibility::LInfinity(a), Sensibility::LInfinity(b)) => Sensibility::LInfinity((compose)((self.value.numel() as f32).sqrt() * a, (other.value.numel() as f32).sqrt() * b)),
                _ => Sensibility::Unknown,
            }
        }, batch_dependence = self.batch_dependence.clone() + other.batch_dependence.clone()
        // f_mse_loss(self: &Self, target: &Self, reduction: Reduction) -> Result<Self, TchError> where sensibility = Sensibility::Unknown, batch_dependence = self.batch_dependence.clone()
        // f_cross_entropy_loss(self: &Self, target: &Self, weight: Option<impl Borrow<Tensor>>, reduction: Reduction, ignore_index: i64, label_smoothing: f64) -> Result<Self, TchError> where sensibility = Sensibility::Unknown, batch_dependence = self.batch_dependence.clone()
    }
}

impl PrivacyGuard<()> {
    pub fn get_private(self, budget: PrivacyBudget) -> Result<(), TchError> {
        match budget {
            PrivacyBudget::NotPrivate => Ok(self.get_non_private()),
            PrivacyBudget::Private(_) => {
                if let Sensibility::Unknown = self.sensibility {
                    return Err(TchError::Kind(String::from(
                        "Unknown sensibility. Consider clipping prior to noising.",
                    )));
                }
                let mut context = self.context.write().unwrap();
                if !context.within_bounds(budget) {
                    return Err(TchError::Kind(String::from("Privacy limit violation.")));
                }
                context.update_budget(budget);
                Ok(())
            }
        }
    }
}
