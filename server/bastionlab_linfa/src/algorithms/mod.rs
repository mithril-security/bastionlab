use linfa::PlattParams;
use linfa_bayes::{GaussianNb, GaussianNbParams};
use linfa_clustering::{KMeans, KMeansInit, KMeansParams};
use linfa_elasticnet::{ElasticNet, ElasticNetParams};
use linfa_kernel::KernelParams;
use linfa_linear::LinearRegression;
use linfa_logistic::LogisticRegression;
use linfa_nn::{
    distance::{Distance, L2Dist},
    BallTreeIndex, BuildError, KdTreeIndex, LinearSearchIndex,
};
use linfa_svm::SvmParams;
use linfa_trees::{DecisionTreeParams, SplitQuality};
use ndarray::{Array1, ArrayBase, Data, Ix2};

pub fn logistic_regression(
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

pub fn gaussian_naive_bayes(var_smoothing: f64) -> GaussianNbParams<f64, usize> {
    let model = GaussianNb::params().var_smoothing(var_smoothing);

    model
}

pub fn linear_regression(fit_intercept: bool) -> LinearRegression {
    let model = LinearRegression::new();
    let model = model.with_intercept(fit_intercept);

    model
}

pub fn decision_trees(
    split_quality: SplitQuality,
    max_depth: Option<usize>,
    min_weight_split: f32,
    min_weight_leaf: f32,
    min_impurity_decrease: f64,
) -> DecisionTreeParams<f64, usize> {
    let reg = DecisionTreeParams::new()
        .split_quality(split_quality)
        .max_depth(max_depth)
        .min_weight_split(min_weight_split)
        .min_weight_leaf(min_weight_leaf)
        .min_impurity_decrease(min_impurity_decrease);

    reg
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
) -> KMeansParams<f64, rand_xoshiro::Xoshiro256Plus, L2Dist> {
    let model = KMeans::params(n_clusters)
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

pub fn svm(
    c: Vec<f64>,
    eps: Option<f64>,
    nu: Option<f64>,
    shrinking: bool,
    platt_params: PlattParams<f64, ()>,
    kernel_params: KernelParams<f64>,
) -> SvmParams<f64, f64> {
    let model = SvmParams::new()
        .shrinking(shrinking)
        .with_platt_params(platt_params)
        .with_kernel_params(kernel_params);

    if c.len() > 0 {
        if c.len() == 2 {
            model.pos_neg_weights(c[0], c[1])
        } else if c.len() == 1 && eps.is_some() {
            model.c_eps(c[0], eps.unwrap())
        } else {
            model
        }
    } else if nu.is_some() && eps.is_some() {
        model.nu_eps(nu.unwrap(), eps.unwrap())
    } else {
        model
    }
}
