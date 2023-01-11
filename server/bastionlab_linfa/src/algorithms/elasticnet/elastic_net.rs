use linfa_elasticnet::{ElasticNet, ElasticNetParams};

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
