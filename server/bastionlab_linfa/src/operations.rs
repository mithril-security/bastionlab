use std::sync::Arc;

use bastionlab_common::array_store::ArrayStore;
use linfa::{
    prelude::{SingleTargetRegression, ToConfusionMatrix},
    traits::{Fit, Predict},
};
use ndarray::{Array, Array1, Array2, ArrayBase, Dimension, Ix1, OwnedRepr, StrideShape, ViewRepr};

use polars::{
    export::{
        arrow::types::PrimitiveType,
        num::{FromPrimitive, ToPrimitive},
    },
    prelude::{
        DataFrame, Float32Chunked, Float64Chunked, Float64Type, NumericNative, PolarsError,
        PolarsResult, UInt32Chunked, UInt64Chunked,
    },
    series::Series,
};
use tonic::Status;

use crate::{
    algorithms::*,
    trainer::{get_datasets, to_polars_error, Models, PredictionTypes, SupportedModels},
};

pub fn ndarray_to_df<T, D: Dimension>(
    arr: &ArrayBase<OwnedRepr<T>, D>,
    col_names: Vec<&str>,
) -> PolarsResult<DataFrame>
where
    T: NumericNative + FromPrimitive + ToPrimitive,
{
    let mut lanes: Vec<Series> = vec![];

    for (i, col) in arr.columns().into_iter().enumerate() {
        match col.as_slice() {
            Some(d) => {
                if let PrimitiveType::Float64 = T::PRIMITIVE {
                    let d = d
                        .into_iter()
                        .map(|v| v.to_f64().unwrap())
                        .collect::<Vec<_>>();
                    let series = Series::from(Float64Chunked::new_vec(col_names[i], d));
                    lanes.push(series);
                }
                if let PrimitiveType::Float32 = T::PRIMITIVE {
                    let d = d
                        .into_iter()
                        .map(|v| v.to_f32().unwrap())
                        .collect::<Vec<_>>();
                    let series = Series::from(Float32Chunked::new_vec(col_names[i], d));
                    lanes.push(series);
                }
                if let PrimitiveType::UInt64 = T::PRIMITIVE {
                    let d = d
                        .into_iter()
                        .map(|v| v.to_u64().unwrap())
                        .collect::<Vec<_>>();
                    let series = Series::from(UInt64Chunked::new_vec(col_names[i], d));
                    lanes.push(series);
                }
                if let PrimitiveType::UInt32 = T::PRIMITIVE {
                    let d = d
                        .into_iter()
                        .map(|v| v.to_u32().unwrap())
                        .collect::<Vec<_>>();
                    let series = Series::from(UInt32Chunked::new_vec("col", d));
                    lanes.push(series);
                }
                // This could be expanded... for now, only (f64,f32, u64, and u32) are supported.
            }
            None => {
                return Err(PolarsError::NoData(polars::error::ErrString::Borrowed(
                    "Could not convert column to slice",
                )));
            }
        }
    }

    DataFrame::new(lanes)
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
    records: ArrayStore,
    targets: ArrayStore,
    model_type: Models,
) -> Result<SupportedModels, Status> {
    let train = get_datasets(records, targets);
    let failed_array_type = |model, array| {
        return Err(Status::failed_precondition(format!(
            "{model} expect array to be of type f64: {:?}",
            array
        )));
    };
    match model_type {
        Models::GaussianNaiveBayes { var_smoothing } => {
            let model = gaussian_naive_bayes(var_smoothing.into());

            let records = match train.records.0 {
                ArrayStore::AxdynF64(a) => a,
                _ => return failed_array_type("Gaussian Naive Bayes -> Records", train.records),
            };
            let targets = match train.targets.0 {
                ArrayStore::AxdynUsize(a) => a,
                _ => return failed_array_type("Gaussian Naive Bayes -> Targets", train.targets),
            };
            todo!()
            // let model = to_polars_error(model.fit(&train))?;
            // Ok(SupportedModels::GaussianNaiveBayes(model))
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
            todo!()
            // Ok(SupportedModels::ElasticNet(to_polars_error(
            //     model.fit(&train),
            // )?))
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
            todo!()
            // Ok(SupportedModels::KMeans(to_polars_error(model.fit(&train))?))
        }
        Models::LinearRegression { fit_intercept } => {
            let model = linear_regression(fit_intercept);

            todo!()
            // Ok(SupportedModels::LinearRegression(to_polars_error(
            //     model.fit(&train),
            // )?))
        }

        Models::LogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
        } => {
            // let targets = to_usize(&train.targets, train.targets.shape().to_vec()[0])?;
            // let train = train.with_targets(targets);
            // let model = logistic_regression(
            //     alpha,
            //     gradient_tolerance,
            //     fit_intercept,
            //     max_iterations,
            //     initial_params,
            // );
            todo!()
            // Ok(SupportedModels::LogisticRegression(to_polars_error(
            //     model.fit(&train),
            // )?))
        }

        Models::DecisionTree {
            split_quality,
            max_depth,
            min_weight_split,
            min_weight_leaf,
            min_impurity_decrease,
        } => {
            // let shape = train.targets.shape().to_vec();
            // let targets = to_usize(&train.targets, shape[0])?;
            // let train = train.with_targets(targets);
            // let model = decision_trees(
            //     split_quality,
            //     max_depth,
            //     min_weight_split,
            //     min_weight_leaf,
            //     min_impurity_decrease,
            // );
            todo!()
            // Ok(SupportedModels::DecisionTree(to_polars_error(
            //     model.fit(&train),
            // )?))
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
            todo!()
            // Ok(SupportedModels::SVM(to_polars_error(model.fit(&train))?))
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
                let targets = match m.targets.as_slice() {
                    Some(s) => {
                        let arr = s.into_iter().map(|v| *v as u64).collect::<Vec<u64>>();
                        Array1::from_vec(arr)
                    }
                    None => {
                        return Err(PolarsError::InvalidOperation(
                            polars::error::ErrString::Borrowed("Could not convert to slice"),
                        ));
                    }
                };
                ndarray_to_df::<u64, Ix1>(&targets, vec!["prediction"])?
            }
            PredictionTypes::Float(m) => ndarray_to_df::<f64, Ix1>(&m.targets, vec!["prediction"])?,
            PredictionTypes::Probability(pr) => ndarray_to_df::<f64, Ix1>(&pr, vec!["prediction"])?,
        },
        None => {
            return PolarsResult::Err(PolarsError::ComputeError(polars::error::ErrString::Owned(
                "Failed to predict".to_string(),
            )))
        }
    };

    Ok(prediction)
}

fn regression_metrics(
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

fn classification_metrics(
    prediction: &ArrayBase<OwnedRepr<usize>, Ix1>,
    truth: &ArrayBase<ViewRepr<&usize>, Ix1>,
    metric: &str,
) -> Result<f32, linfa::Error> {
    let cm = prediction.confusion_matrix(truth)?;
    match metric {
        "accuracy" => Ok(cm.accuracy()),
        "f1_score" => Ok(cm.f1_score()),
        "mcc" => Ok(cm.mcc()),
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
    todo!()
    // let (cols, records, targets) = transform(&records, &targets)?;
    // let mut train = get_datasets(records, targets);

    // let result = match model {
    //     Models::LinearRegression { fit_intercept } => {
    //         let m = linear_regression(fit_intercept);

    //         let arr = to_polars_error(train.cross_validate_single(
    //             cv,
    //             &vec![m][..],
    //             |pred, truth| regression_metrics(pred, truth, scoring),
    //         ))?;

    //         ndarray_to_df::<f64, Ix1>(&arr, vec![scoring])
    //     }

    //     Models::LogisticRegression {
    //         alpha,
    //         gradient_tolerance,
    //         fit_intercept,
    //         max_iterations,
    //         initial_params,
    //     } => {
    //         let m = logistic_regression(
    //             alpha,
    //             gradient_tolerance,
    //             fit_intercept,
    //             max_iterations,
    //             initial_params,
    //         );

    //         let targets = to_usize(&train.targets, train.targets.shape().to_vec()[0])?;
    //         let mut train = train.with_targets(targets);
    //         let arr = to_polars_error(train.cross_validate_single(
    //             cv,
    //             &vec![m][..],
    //             |pred, truth| classification_metrics(pred, truth, scoring),
    //         ))?;

    //         println!("{:?}", arr);

    //         ndarray_to_df::<f32, Ix1>(&arr, vec![scoring])
    //     }

    //     _ => {
    //         return Err(PolarsError::NotFound(polars::error::ErrString::Owned(
    //             "Unsupported Model".to_owned(),
    //         )))
    //     }
    // };

    // result
}
