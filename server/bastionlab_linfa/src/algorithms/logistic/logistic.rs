use linfa::{traits::Fit, DatasetBase};
use linfa_logistic::{error::Result, FittedLogisticRegression, LogisticRegression};
use ndarray::{Array1, ArrayBase, Ix1, Ix2, OwnedRepr};

pub fn logistic_regression(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>,
    alpha: f64,
    gradient_tolerance: f64,
    fit_intercept: bool,
    max_iterations: u64,
    inital_params: Option<Vec<f64>>,
) -> Result<FittedLogisticRegression<f64, usize>> {
    let mut reg = LogisticRegression::new()
        .alpha(alpha)
        .with_intercept(fit_intercept)
        .gradient_tolerance(gradient_tolerance)
        .max_iterations(max_iterations);

    if let Some(params) = inital_params {
        let params = Array1::from_vec(params);
        reg = reg.initial_params(params);
    }

    let model = reg.fit(&train)?;

    println!("{:?}", model);
    Ok(model)
}
