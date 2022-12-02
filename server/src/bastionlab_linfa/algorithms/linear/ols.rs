use linfa::{traits::Fit, DatasetBase};
use linfa_linear::{FittedLinearRegression, LinearError, LinearRegression};
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr};

pub fn linear_regression(
    dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>,
    fit_intercept: bool,
) -> Result<FittedLinearRegression<f64>, LinearError<f64>> {
    let model = LinearRegression::new();
    let model = model.with_intercept(fit_intercept).fit(&dataset)?;

    println!("{:?}", model);
    Ok(model)
}
