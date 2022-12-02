use linfa::traits::Predict;
use ndarray::Array2;

use polars::{
    error::ErrString,
    prelude::{DataFrame, NamedFrom, PolarsError, PolarsResult},
    series::Series,
};
use tonic::Status;

use crate::bastionlab_linfa::{
    algorithms::{elastic_net, gaussian_naive_bayes, kmeans, linear_regression},
    to_status_error,
    trainer::{get_datasets, to_polars_error, Models, PredictionTypes, SupportedModels},
};

#[macro_export]
macro_rules! to_type {
    {<$t_ty:ty>($item:ident)} => {
        // Use specified type.
        $item.iter().map(|v| *v as $t_ty).collect::<Vec<$t_ty>>()
    };
}

#[macro_export]
macro_rules! to_ndarray {
    ($sh:ident, $target:ident) => {
        to_polars_error(Array2::from_shape_vec($sh, $target))?
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

pub fn predict(model: SupportedModels, data: Vec<f64>) -> PolarsResult<Option<PredictionTypes>> {
    let sh = (1, data.len());
    let sample = to_ndarray!(sh, data);
    let prediction = match model {
        SupportedModels::ElasticNet(m) => Some(PredictionTypes::Float(m.predict(sample))),
        SupportedModels::GaussianNaiveBayes(m) => Some(PredictionTypes::Usize(m.predict(sample))),

        SupportedModels::KMeans(m) => Some(PredictionTypes::Usize(m.predict(sample))),
        SupportedModels::LinearRegression(m) => Some(PredictionTypes::Float(m.predict(sample))),
    };
    Ok(prediction)
}

pub fn send_to_trainer(
    records: DataFrame,
    target: DataFrame,
    ratio: f32,
    model_type: Models,
) -> PolarsResult<SupportedModels> {
    // We are assuming [`f64`] for all computation since it can represent all other types.
    match model_type {
        Models::GaussianNaiveBayes { var_smoothing } => {
            let var_smoothing: f64 = var_smoothing.into();
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<usize>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = target
                .clone()
                .into_shape([target.clone().len()])
                .map_err(|e| PolarsError::InvalidOperation(ErrString::Owned(e.to_string())))?;

            let records = to_ndarray!(records_shape, records);

            let (train, valid) = get_datasets(records, target, ratio, cols)?;
            let (model, _) =
                to_polars_error(gaussian_naive_bayes(train, valid, var_smoothing.into()))?;
            Ok(SupportedModels::GaussianNaiveBayes(model))
        }
        Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<f64>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (train, valid) = get_datasets(records, target, ratio, cols)?;
            let model = to_polars_error(elastic_net(
                train,
                valid,
                penalty.into(),
                l1_ratio.into(),
                with_intercept,
                max_iterations,
                tolerance.into(),
            ))?;
            Ok(SupportedModels::ElasticNet(model))
        }
        Models::KMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
        } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<f64>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (train, valid) = get_datasets(records, target, ratio, cols)?;
            let model = to_polars_error(kmeans(
                train,
                valid,
                n_runs.into(),
                n_clusters.into(),
                tolerance,
                max_n_iterations,
                init_method,
            ))?;
            Ok(SupportedModels::KMeans(model))
        }
        Models::LinearRegression { fit_intercept } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<f64>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (dataset, _) = get_datasets(records, target, ratio, cols)?;
            let model = to_polars_error(linear_regression(dataset, fit_intercept))?;

            Ok(SupportedModels::LinearRegression(model))
        }
    }
}

pub fn get_prediction(pred: Option<PredictionTypes>) -> Result<DataFrame, Status> {
    let prediction: DataFrame = match pred {
        Some(v) => match v {
            PredictionTypes::Usize(m) => {
                let targets = m.targets.to_vec();
                let targets = to_type! {<u64>(targets)};
                let s = Series::new("prediction", targets);
                let df = to_status_error(DataFrame::new(vec![s]))?;
                df
            }
            PredictionTypes::Float(m) => {
                let targets = m.targets.to_vec();
                let targets = to_type! {<f64>(targets)};
                let s = Series::new("prediction", targets);
                let df = to_status_error(to_polars_error(DataFrame::new(vec![s])))?;
                df
            }
        },
        None => {
            return Err(Status::aborted("Failed to predict!"));
        }
    };

    Ok(prediction)
}
