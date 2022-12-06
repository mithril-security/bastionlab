use linfa::{
    prelude::{ConfusionMatrix, ToConfusionMatrix},
    traits::{Fit, Predict},
    DatasetBase,
};
use linfa_bayes::{GaussianNb, Result};
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr};

pub fn gaussian_naive_bayes(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>,
    valid: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>,
    var_smoothing: f64,
) -> Result<(GaussianNb<f64, usize>, ConfusionMatrix<usize>)> {
    let model = GaussianNb::params()
        .var_smoothing(var_smoothing)
        .fit(&train)?;

    let pred = model.predict(&valid);

    let cm = pred.confusion_matrix(&valid)?;

    Ok((model, cm))
}
