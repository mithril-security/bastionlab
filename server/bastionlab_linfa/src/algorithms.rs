use linfa_bayes::{GaussianNb, GaussianNbParams};
use linfa_clustering::{KMeans, KMeansInit, KMeansParams};
use linfa_elasticnet::{ElasticNet, ElasticNetParams};
use linfa_linear::{LinearRegression, Link, TweedieRegressor, TweedieRegressorParams};
use linfa_logistic::{LogisticRegression, MultiLogisticRegression};
use linfa_nn::{
    distance::{Distance, L2Dist},
    BallTreeIndex, BuildError, KdTreeIndex, LinearSearchIndex,
};
use linfa_svm::SvmParams;
use linfa_trees::{DecisionTreeParams, SplitQuality};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use rand::{rngs::StdRng, SeedableRng};
use tonic::Status;

use crate::utils::LabelU64;

pub fn binomial_logistic_regression(
    alpha: f64,
    gradient_tolerance: f64,
    fit_intercept: bool,
    max_iterations: u64,
    initial_params: Option<Vec<f64>>,
) -> LogisticRegression<f64> {
    let mut reg = LogisticRegression::new()
        .alpha(alpha)
        .with_intercept(fit_intercept)
        .gradient_tolerance(gradient_tolerance)
        .max_iterations(max_iterations);

    if let Some(params) = initial_params {
        let params = Array1::from_vec(params);
        reg = reg.initial_params(params);
    }

    reg
}

pub fn multinomial_logistic_regression(
    alpha: f64,
    gradient_tolerance: f64,
    fit_intercept: bool,
    max_iterations: u64,
    initial_params: Option<Vec<f64>>,
    shape: Option<Vec<u64>>,
) -> Result<MultiLogisticRegression<f64>, Status> {
    let mut reg = MultiLogisticRegression::new()
        .alpha(alpha)
        .with_intercept(fit_intercept)
        .gradient_tolerance(gradient_tolerance)
        .max_iterations(max_iterations);

    if let Some(params) = initial_params {
        if let Some(shape) = shape {
            let params = Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), params)
                .map_err(|e| {
                    Status::failed_precondition(format!(
                        "Could not construct initial_params from shape: {:?}: {e}",
                        shape
                    ))
                })?;
            reg = reg.initial_params(params);
        }
    }

    Ok(reg)
}

pub fn gaussian_naive_bayes(var_smoothing: f64) -> GaussianNbParams<f64, LabelU64> {
    let model = GaussianNb::params().var_smoothing(var_smoothing);

    model
}

pub fn linear_regression(fit_intercept: bool) -> LinearRegression {
    let model = LinearRegression::new();
    let model = model.with_intercept(fit_intercept);
    model
}

pub fn tweedie_regression(
    fit_intercept: bool,
    alpha: f64,
    max_iter: u64,
    link: Link,
    tol: f64,
    power: f64,
) -> TweedieRegressorParams<f64> {
    let model = TweedieRegressor::params()
        .fit_intercept(fit_intercept)
        .alpha(alpha)
        .max_iter(max_iter.try_into().unwrap())
        .link(link)
        .tol(tol)
        .power(power);

    model
}

pub fn decision_trees(
    split_quality: SplitQuality,
    max_depth: Option<usize>,
    min_weight_split: f32,
    min_weight_leaf: f32,
    min_impurity_decrease: f64,
) -> DecisionTreeParams<f64, LabelU64> {
    let model = DecisionTreeParams::new()
        .split_quality(split_quality)
        .max_depth(max_depth)
        .min_weight_split(min_weight_split)
        .min_weight_leaf(min_weight_leaf)
        .min_impurity_decrease(min_impurity_decrease);

    model
}

pub fn elastic_net(
    penalty: f64,
    l1_ratio: f64,
    with_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
) -> ElasticNetParams<f64> {
    let model = ElasticNet::params()
        .penalty(penalty)
        .l1_ratio(l1_ratio)
        .with_intercept(with_intercept)
        .max_iterations(max_iterations)
        .tolerance(tolerance);
    model
}

pub fn kmeans(
    n_runs: usize,
    n_clusters: usize,
    tolerance: f64,
    max_n_iterations: u64,
    init_method: KMeansInit<f64>,
    random_state: u64,
) -> KMeansParams<f64, StdRng, L2Dist> {
    let rng = StdRng::seed_from_u64(random_state);
    let model = KMeans::params_with(n_clusters, rng, L2Dist)
        .n_runs(n_runs)
        .max_n_iterations(max_n_iterations)
        .tolerance(tolerance)
        .init_method(init_method);

    model
}

#[allow(unused)]
pub fn linear_search<DT: Data<Elem = f64>, D: Distance<f64>>(
    batch: &ArrayBase<DT, Ix2>,
    distance: D,
) -> Result<LinearSearchIndex<f64, D>, BuildError> {
    LinearSearchIndex::new(batch, distance)
}

#[allow(unused)]
pub fn kdtree<DT: Data<Elem = f64>, D: Distance<f64>>(
    batch: &ArrayBase<DT, Ix2>,
    leaf_size: usize,
    dist_fn: D,
) -> Result<KdTreeIndex<f64, D>, BuildError> {
    KdTreeIndex::new(batch, leaf_size, dist_fn)
}

#[allow(unused)]
pub fn balltree<DT: Data<Elem = f64>, D: Distance<f64>>(
    batch: &ArrayBase<DT, Ix2>,
    leaf_size: usize,
    dist_fn: D,
) -> Result<BallTreeIndex<f64, D>, BuildError> {
    BallTreeIndex::new(batch, leaf_size, dist_fn)
}

#[allow(unused)]
pub fn svm(
    c: f64,
    eps: Option<f64>,
    shrinking: bool,
    gamma: String,
    degree: u64,
    max_iter: i64,
    coef0: f64,
    kernel: String,
) -> SvmParams<f64, f64> {
    todo!()
}
