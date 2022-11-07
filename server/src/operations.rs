use ndarray::Array2;
use polars::{
    error::ErrString,
    prelude::{DataFrame, PolarsError, PolarsResult},
};

use crate::trainer::{
    elastic_net, gaussian_naive_bayes, get_datasets, to_polars_error, Models, SupportedModels,
};

macro_rules! to_type {
    {<$t_ty:ty>($item:ident)} => {
        // Use specified type.
        $item.iter().map(|v| *v as $t_ty).collect::<Vec<$t_ty>>()
    };
}

macro_rules! to_ndarray {
    ($sh:ident, $target:ident) => {
        Array2::from_shape_vec($sh, $target).unwrap()
    };
}

fn df_to_vec(df: DataFrame, sh: (usize, usize)) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(sh.0 * sh.1);
    let df = df.get_columns();
    df.iter().for_each(|s| {
        let s = s.cast(&polars::prelude::DataType::Float64).unwrap();
        let s = s.f64().unwrap();
        let s = s.cont_slice().unwrap();
        out.append(&mut s.to_vec());
    });
    out
}

struct Trainer {
    records: Vec<f64>,
    target: Vec<f64>,
    cols: Vec<String>,
    records_shape: (usize, usize),
    target_shape: (usize, usize),
}

fn transform_dfs(records: DataFrame, target: DataFrame) -> Trainer {
    let cols = records
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let (records_shape, target_shape) = (records.shape(), target.shape());
    let records = df_to_vec(records.clone(), records.shape());
    let target = df_to_vec(target.clone(), target.shape());

    Trainer {
        records,
        target,
        cols,
        records_shape,
        target_shape,
    }
}

pub fn send_to_trainer(
    records: DataFrame,
    target: DataFrame,
    ratio: f32,
    model_type: Models,
) -> PolarsResult<SupportedModels> {
    // We are assuming [`f64`] for all computation since it can represent all other types.
    match model_type {
        Models::GaussianNaiveBayes => {
            let trainer = transform_dfs(records, target);

            let records = trainer.records;
            let target = trainer.target;
            let col_names = trainer.cols;
            let r_shape = trainer.records_shape;
            let t_shape = trainer.target_shape;

            let target = to_type! {<usize>(target)};

            let target = to_ndarray!(t_shape, target);
            let target = target
                .clone()
                .into_shape([target.clone().len()])
                .map_err(|e| PolarsError::InvalidOperation(ErrString::Owned(e.to_string())))?;

            let records = to_ndarray!(r_shape, records);

            let (train, valid) = get_datasets(records, target, ratio, col_names)?;
            let (model, _) = gaussian_naive_bayes(train, valid)?;
            Ok(SupportedModels::GaussianNaiveBayes(model))
        }
        Models::ElasticNet => {
            let trainer = transform_dfs(records, target);

            let records = trainer.records;
            let target = trainer.target;
            let col_names = trainer.cols;
            let r_shape = trainer.records_shape;
            let t_shape = trainer.target_shape;

            let target = to_type! {<f64>(target)};

            let target = to_ndarray!(t_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(r_shape, records);
            let (train, valid) = get_datasets(records, target, ratio, col_names)?;
            let model = elastic_net(train, valid)?;
            Ok(SupportedModels::ElasticNet(model))
        }
    }
}
