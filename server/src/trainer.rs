use std::error::Error;

use linfa::{
    prelude::{ConfusionMatrix, SingleTargetRegression, ToConfusionMatrix},
    traits::{Fit, Predict},
    DatasetBase,
};
use linfa_bayes::GaussianNb;
use linfa_elasticnet::ElasticNet;
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr};
use polars::{
    error::ErrString,
    prelude::{PolarsError, PolarsResult},
};

pub enum Models {
    GaussianNaiveBayes,
    ElasticNet,
}

#[derive(Debug, Clone)]
pub enum SupportedModels {
    GaussianNaiveBayes(GaussianNb<f64, usize>),
    ElasticNet(ElasticNet<f64>),
}

pub fn to_polars_error<T, E: Error>(input: Result<T, E>) -> PolarsResult<T> {
    input.map_err(|err| PolarsError::InvalidOperation(ErrString::Owned(err.to_string())))
}

pub fn get_datasets<D, T>(
    records: ArrayBase<OwnedRepr<D>, Ix2>,
    target: ArrayBase<OwnedRepr<T>, Ix1>,
    ratio: f32,
    col_names: Vec<String>,
) -> PolarsResult<(
    DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, Ix1>>,
    DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, Ix1>>,
)> {
    let dataset = linfa::Dataset::new(records, target);

    let dataset = dataset.with_feature_names::<String>(col_names);

    let (train_set, test_set) = dataset.split_with_ratio(ratio);
    Ok((train_set, test_set))
}

pub fn gaussian_naive_bayes(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>,
    valid: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>,
) -> PolarsResult<(GaussianNb<f64, usize>, ConfusionMatrix<usize>)> {
    let model = to_polars_error(GaussianNb::params().fit(&train))?;
    let pred = model.predict(&valid);

    let cm = to_polars_error(pred.confusion_matrix(&valid))?;

    Ok((model, cm))
}

pub fn elastic_net(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>,
    valid: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>,
) -> PolarsResult<ElasticNet<f64>> {
    let model = to_polars_error(ElasticNet::params().penalty(0.1).l1_ratio(1.0).fit(&train))?;
    let y_est = model.predict(&valid);
    println!("predicted : {}", to_polars_error(y_est.r2(&valid))?);
    Ok(model)
}
