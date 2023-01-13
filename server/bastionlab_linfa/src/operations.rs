use std::sync::Arc;

use linfa::{
    prelude::SingleTargetRegression,
    traits::{Fit, Predict},
};
use ndarray::{Array, Array1, Array2, ArrayBase, Dimension, Ix1, OwnedRepr, StrideShape, ViewRepr};

use polars::{
    prelude::{DataFrame, Float64Type, NamedFrom, PolarsError, PolarsResult},
    series::Series,
};

use crate::{
    algorithms::*,
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

fn to_usize<D: Dimension, S: Into<StrideShape<D>>>(
    targets: &ArrayBase<OwnedRepr<f64>, D>,
    shape: S,
) -> PolarsResult<ArrayBase<OwnedRepr<usize>, D>> {
    let targets = match targets.as_slice() {
        Some(s) => {
            let cast = s.into_iter().map(|v| *v as usize).collect::<Vec<_>>();
            to_polars_error(Array::from_shape_vec(shape, cast))?
        }
        None => {
            return Err(PolarsError::InvalidOperation(
                polars::error::ErrString::Borrowed("Could not create slice from targets"),
            ));
        }
    };
    Ok(targets)
}

fn transform(
    records: &DataFrame,
    targets: &DataFrame,
) -> PolarsResult<(Vec<String>, Array2<f64>, Array1<f64>)> {
    let cols = records
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let targets_shape = targets.shape();
    let records = to_polars_error(records.to_ndarray::<Float64Type>())?
        .as_standard_layout()
        .to_owned();
    let targets = to_polars_error(targets.to_ndarray::<Float64Type>())?
        .as_standard_layout()
        .to_owned();
    let targets = to_polars_error(targets.into_shape(targets_shape.0))?;

    Ok((cols, records, targets))
}

/// This method sends both the training and target datasets to the specified model in [`Models`].
/// And `ratio` is passed along to [`linfa_datasets::DatasetBase`]
pub fn send_to_trainer(
    records: DataFrame,
    targets: DataFrame,
    model_type: Models,
) -> PolarsResult<SupportedModels> {
    let (cols, records, targets) = transform(&records, &targets)?;
    let targets_shape = targets.shape().to_vec();

    let train = get_datasets(records, targets, cols)?;

    match model_type {
        Models::GaussianNaiveBayes { var_smoothing } => {
            let targets = to_usize(&train.targets, targets_shape[0])?;
            let train = train.with_targets(targets);
            let model = gaussian_naive_bayes(var_smoothing.into());

            let model = to_polars_error(model.fit(&train))?;
            Ok(SupportedModels::GaussianNaiveBayes(model))
        }
        Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        } => {
            let model = elastic_net(
                penalty.into(),
                l1_ratio.into(),
                with_intercept,
                max_iterations,
                tolerance.into(),
            );
            Ok(SupportedModels::ElasticNet(to_polars_error(
                model.fit(&train),
            )?))
        }
        Models::KMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
        } => {
            let model = kmeans(
                n_runs.into(),
                n_clusters.into(),
                tolerance,
                max_n_iterations,
                init_method,
            );

            Ok(SupportedModels::KMeans(to_polars_error(model.fit(&train))?))
        }
        Models::LinearRegression { fit_intercept } => {
            let model = linear_regression(fit_intercept);

            Ok(SupportedModels::LinearRegression(to_polars_error(
                model.fit(&train),
            )?))
        }

        Models::LogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
        } => {
            let targets = to_usize(&train.targets, train.targets.shape().to_vec()[0])?;
            let train = train.with_targets(targets);
            let model = logistic_regression(
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
            );

            Ok(SupportedModels::LogisticRegression(to_polars_error(
                model.fit(&train),
            )?))
        }

        Models::DecisionTree {
            split_quality,
            max_depth,
            min_weight_split,
            min_weight_leaf,
            min_impurity_decrease,
        } => {
            let shape = train.targets.shape().to_vec();
            let targets = to_usize(&train.targets, shape[0])?;
            let train = train.with_targets(targets);
            let model = decision_trees(
                split_quality,
                max_depth,
                min_weight_split,
                min_weight_leaf,
                min_impurity_decrease,
            );

            Ok(SupportedModels::DecisionTree(to_polars_error(
                model.fit(&train),
            )?))
        }
        Models::SVM {
            c,
            eps,
            nu,
            shrinking,
            platt_params,
            kernel_params,
        } => {
            let c = c
                .iter()
                .map(|v| (*v).try_into().unwrap())
                .collect::<Vec<_>>();
            let eps = match eps {
                Some(v) => Some(v.into()),
                None => None,
            };

            let nu = match nu {
                Some(v) => Some(v.into()),
                None => None,
            };
            let model = svm(c, eps, nu, shrinking, platt_params, kernel_params);

            Ok(SupportedModels::SVM(to_polars_error(model.fit(&train))?))
        }
    }
}

/// This method is used to run a prediction on an already fitted model, based on the model selection type.
/// We use two different types for prediction
/// [f64] and [usize] --> [PredictionTypes::Float] and [PredictionTypes::Usize] respectively.
pub fn predict(
    model: Arc<SupportedModels>,
    data: DataFrame,
    probability: bool,
) -> PolarsResult<DataFrame> {
    let sample = to_polars_error(data.to_ndarray::<Float64Type>())?
        .as_standard_layout()
        .to_owned();

    let prediction = match &*model {
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
        _ => {
            return Err(PolarsError::NotFound(polars::error::ErrString::Borrowed(
                "Unsupported Model",
            )))
        }
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

pub fn metric(
    prediction: &ArrayBase<OwnedRepr<f64>, Ix1>,
    truth: &ArrayBase<ViewRepr<&f64>, Ix1>,
    metric: &str,
) -> Result<f64, linfa::Error> {
    match metric {
        "r2" => prediction.r2(truth),
        "max_error" => prediction.max_error(truth),
        "mean_absolute_error" => prediction.mean_absolute_error(truth),
        "explained_variance" => prediction.explained_variance(truth),
        "mean_squared_log_error" => prediction.mean_squared_log_error(truth),
        "mean_squared_error" => prediction.mean_squared_error(truth),
        "median_absolute_error" => prediction.median_absolute_error(truth),
        _ => {
            return Err(linfa::Error::Priors(format!(
                "Could not find metric: {}",
                metric
            )))
        }
    }
}

#[allow(unused)]
pub fn inner_cross_validate(
    model: Models,
    records: DataFrame,
    targets: DataFrame,
    scoring: &str,
    cv: usize,
) -> PolarsResult<DataFrame> {
    let (cols, records, targets) = transform(&records, &targets)?;
    let mut train = get_datasets(records, targets, cols)?;

    let result = match model {
        Models::LinearRegression { fit_intercept } => {
            let m = linear_regression(fit_intercept);

            let arr = to_polars_error(train.cross_validate_single(
                cv,
                &vec![m][..],
                |pred, truth| metric(pred, truth, scoring),
            ))?;

            let arr = match arr.as_slice() {
                Some(d) => d.to_vec(),
                None => {
                    return Err(PolarsError::InvalidOperation(
                        polars::error::ErrString::Borrowed(
                            "Failed to convert validation result to slice",
                        ),
                    ));
                }
            };
            vec_f64_to_df(arr, scoring)
        }

        _ => {
            return Err(PolarsError::NotFound(polars::error::ErrString::Owned(
                "Unsupported Model".to_owned(),
            )))
        }
    };

    result
}
