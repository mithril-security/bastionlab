use linfa::{traits::Fit, DatasetBase};
use linfa_elasticnet::{ElasticNet, Result};
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr};

pub fn elastic_net(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>,
    penalty: f64,
    l1_ratio: f64,
    with_intercept: bool,
    max_iterations: u32,
    tolerance: f64,
) -> Result<ElasticNet<f64>> {
    let model = ElasticNet::params()
        .penalty(penalty)
        .l1_ratio(l1_ratio)
        .with_intercept(with_intercept)
        .max_iterations(max_iterations)
        .tolerance(tolerance)
        .fit(&train)?;
    Ok(model)
}
