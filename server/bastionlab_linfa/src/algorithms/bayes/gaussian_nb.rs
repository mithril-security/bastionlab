use linfa::DatasetBase;
use linfa_bayes::{GaussianNb, GaussianNbParams};

pub fn gaussian_naive_bayes(var_smoothing: f64) -> GaussianNbParams<f64, usize> {
    let model = GaussianNb::params().var_smoothing(var_smoothing);

    model
}
