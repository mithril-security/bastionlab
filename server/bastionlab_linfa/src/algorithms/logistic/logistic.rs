use linfa::traits::Fit;
use linfa_logistic::{error::Result, FittedLogisticRegression, LogisticRegression};
use ndarray::Array1;

pub fn logistic_regression(
    alpha: f64,
    gradient_tolerance: f64,
    fit_intercept: bool,
    max_iterations: u64,
    inital_params: Option<Vec<f64>>,
) -> LogisticRegression<f64> {
    let mut reg = LogisticRegression::new()
        .alpha(alpha)
        .with_intercept(fit_intercept)
        .gradient_tolerance(gradient_tolerance)
        .max_iterations(max_iterations);

    if let Some(params) = inital_params {
        let params = Array1::from_vec(params);
        reg = reg.initial_params(params);
    }

    reg
}
