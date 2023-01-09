use linfa::traits::Predict;
use ndarray::{Array1, Array2};

use polars::{
    df,
    prelude::{DataFrame, Float64Type, NamedFrom, PolarsError, PolarsResult},
    series::Series,
};
use tonic::Status;

use crate::{
    algorithms::{decision_trees, elastic_net, gaussian_naive_bayes, kmeans, linear_regression},
    to_status_error,
    trainer::{get_datasets, to_polars_error, Models, PredictionTypes, SupportedModels},
};

use super::algorithms::logistic_regression;

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
            .as_standard_layout()
            .to_owned()
    };
}

fn vec_f64_to_df(data: Vec<f64>, name: &str) -> PolarsResult<DataFrame> {
    let s = Series::new(name, data);
    let df = to_polars_error(DataFrame::new(vec![s]))?;
    Ok(df)
}

fn vec_u64_to_df(data: Vec<usize>, name: &str) -> PolarsResult<DataFrame> {
    let data = to_type! {<u64>(data)};
    let s = Series::new(name, data);
    let df = to_polars_error(DataFrame::new(vec![s]))?;
    Ok(df)
}

/// This method sends both the training and target datasets to the specified model in [`Models`].
/// And `ratio` is passed along to [`linfa_datasets::DatasetBase`]
pub fn send_to_trainer(
    records: DataFrame,
    targets: DataFrame,
    ratio: f32,
    model_type: Models,
) -> PolarsResult<(SupportedModels, (DataFrame, DataFrame))> {
    // We are assuming [`f64`] for all computation since it can represent all other types.

    let transform = |records: &DataFrame,
                     targets: &DataFrame|
     -> PolarsResult<(Vec<String>, Array2<f64>, Array1<f64>)> {
        let cols = records
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let shape = targets.shape();
        let records = to_polars_error(records.to_ndarray::<Float64Type>())?
            .as_standard_layout()
            .to_owned();
        let targets = to_polars_error(targets.to_ndarray::<Float64Type>())?
            .as_standard_layout()
            .to_owned();
        let targets = to_polars_error(targets.into_shape(shape.0))?;

        Ok((cols, records, targets))
    };

    let to_usize = |targets: Array1<f64>| -> PolarsResult<Array1<usize>> {
        let targets = match targets.as_slice() {
            Some(s) => {
                let cast = s.into_iter().map(|v| *v as usize).collect::<Vec<_>>();
                Array1::from_vec(cast)
            }
            None => {
                return Err(PolarsError::InvalidOperation(
                    polars::error::ErrString::Borrowed("Could not create slice from targets"),
                ));
            }
        };
        Ok(targets)
    };

    match model_type {
        Models::GaussianNaiveBayes { var_smoothing } => {
            let var_smoothing: f64 = var_smoothing.into();

            let (cols, records, targets) = transform(&records, &targets)?;
            let targets = to_usize(targets)?;

            let (train, test) = get_datasets(records, targets, ratio, cols)?;
            let (model, _) = to_polars_error(gaussian_naive_bayes(
                train,
                test.clone(),
                var_smoothing.into(),
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_u64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::GaussianNaiveBayes(model), (rec, tar)))
        }
        Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        } => {
            let (cols, records, targets) = transform(&records, &targets)?;
            let (train, test) = get_datasets(records, targets, ratio, cols)?;
            let model = to_polars_error(elastic_net(
                train,
                test.clone(),
                penalty.into(),
                l1_ratio.into(),
                with_intercept,
                max_iterations,
                tolerance.into(),
            ))?;
            let rec = vec_f64_to_df(test.records.clone().into_raw_vec(), "test_recs")?;
            let tar = vec_f64_to_df(test.targets.clone().into_raw_vec(), "test_target")?;
            Ok((SupportedModels::ElasticNet(model), (rec, tar)))
        }
        Models::KMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
        } => {
            let (cols, records, targets) = transform(&records, &targets)?;
            let (train, test) = get_datasets(records, targets, ratio, cols)?;
            let model = to_polars_error(kmeans(
                train,
                test.clone(),
                n_runs.into(),
                n_clusters.into(),
                tolerance,
                max_n_iterations,
                init_method,
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_f64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::KMeans(model), (rec, tar)))
        }
        Models::LinearRegression { fit_intercept } => {
            let (cols, records, targets) = transform(&records, &targets)?;
            let (dataset, test) = get_datasets(records, targets, ratio, cols)?;
            let model = to_polars_error(linear_regression(dataset, fit_intercept))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_f64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::LinearRegression(model), (rec, tar)))
        }

        Models::LogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
        } => {
            let (cols, records, targets) = transform(&records, &targets)?;
            let targets = to_usize(targets)?;
            let (dataset, test) = get_datasets(records, targets, ratio, cols)?;

            let model = to_polars_error(logistic_regression(
                dataset,
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_u64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::LogisticRegression(model), (rec, tar)))
        }

        Models::DecisionTree {
            split_quality,
            max_depth,
            min_weight_split,
            min_weight_leaf,
            min_impurity_decrease,
        } => {
            let (cols, records, targets) = transform(&records, &targets)?;
            let targets = to_usize(targets)?;

            let (train, test) = get_datasets(records, targets, ratio, cols)?;

            let model = to_polars_error(decision_trees(
                train,
                split_quality,
                max_depth,
                min_weight_split,
                min_weight_leaf,
                min_impurity_decrease,
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_u64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::DecisionTree(model), (rec, tar)))
        }
    }
}

/// This method is used to run a prediction on an already fitted model, based on the model selection type.
/// We use two different types for prediction
/// [f64] and [usize] --> [PredictionTypes::Float] and [PredictionTypes::Usize] respectively.
pub fn predict(
    model: SupportedModels,
    data: Vec<f64>,
    probability: bool,
) -> PolarsResult<DataFrame> {
    let sh = (1, data.len());
    let sample = to_ndarray!(sh, data);
    let prediction = match model {
        SupportedModels::ElasticNet(m) => Some(PredictionTypes::Float(m.predict(sample))),
        SupportedModels::GaussianNaiveBayes(m) => Some(PredictionTypes::Usize(m.predict(sample))),
        SupportedModels::KMeans(m) => Some(PredictionTypes::Usize(m.predict(sample))),
        SupportedModels::LinearRegression(m) => Some(PredictionTypes::Float(m.predict(sample))),
        SupportedModels::LogisticRegression(m) => {
            if probability {
                Some(PredictionTypes::Probability(
                    m.predict_probabilities(&sample),
                ))
            } else {
                Some(PredictionTypes::Usize(m.predict(sample)))
            }
        }
        SupportedModels::DecisionTree(m) => Some(PredictionTypes::Usize(m.predict(sample))),
    };

    let prediction: DataFrame = match prediction {
        Some(v) => match v {
            PredictionTypes::Usize(m) => {
                let targets = m.targets.to_vec();
                vec_u64_to_df(targets, "prediction")?
            }
            PredictionTypes::Float(m) => {
                let targets = m.targets.to_vec();
                vec_f64_to_df(targets, "prediction")?
            }
            PredictionTypes::Probability(m) => vec_f64_to_df(m.to_vec(), "prediction")?,
        },
        None => {
            return PolarsResult::Err(PolarsError::ComputeError(polars::error::ErrString::Owned(
                "Failed to predict".to_string(),
            )))
        }
    };

    Ok(prediction)
}

#[allow(unused)]
pub fn inner_cross_validate(
    model: &SupportedModels,
    test_set: (DataFrame, DataFrame),
) -> Result<DataFrame, Status> {
    let result = match model {
        SupportedModels::LinearRegression(m) => (),
        SupportedModels::LogisticRegression(m) => (),
        SupportedModels::ElasticNet(m) => (),
        SupportedModels::DecisionTree(m) => (),
        SupportedModels::KMeans(m) => (),
        SupportedModels::GaussianNaiveBayes(m) => (),
    };

    let df = to_status_error(df!("scores" => &[0]));
    df
}
