use std::sync::Arc;

use bastionlab_common::{
    array_store::{ArrayStore, ArrayStoreType},
    common_conversions::to_status_error,
};
use linfa::{
    traits::{Fit, Predict},
    DatasetBase,
};
use ndarray::{Array2, Ix2};

use tonic::Status;

use crate::{
    algorithms::*,
    get_inner_array,
    linfa_proto::validation_request::Scoring,
    prepare_train_data,
    trainers::{Models, SupportedModels},
    utils::{get_datasets, IArrayStore, LabelU64},
};

#[allow(unused)]
/// This method sends both the training and target datasets to the specified model in [`Models`].
pub fn send_to_trainer(
    records: ArrayStore,
    targets: ArrayStore,
    model_type: Models,
) -> Result<SupportedModels, Status> {
    let train = get_datasets(records.clone(), targets.clone());

    match model_type {
        Models::GaussianNaiveBayes { var_smoothing } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::UInt64)?));
            let train = prepare_train_data! {
                "GaussianNaiveBayes", train, (AxdynU64, Ix1)
            };

            let train = train.map_targets(|t| LabelU64(*t));

            let model = gaussian_naive_bayes(var_smoothing.into());
            Ok(SupportedModels::GaussianNaiveBayes(to_status_error(
                model.fit(&train),
            )?))
        }
        Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::Float64)?));

            let train = prepare_train_data! {"ElasticNet", train,  (AxdynF64, Ix1) };
            let model = elastic_net(
                penalty.into(),
                l1_ratio.into(),
                with_intercept,
                max_iterations,
                tolerance.into(),
            );
            Ok(SupportedModels::ElasticNet(to_status_error(
                model.fit(&train),
            )?))
        }
        Models::KMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
            random_state,
        } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::Float64)?));
            let train = prepare_train_data! {"KMeans", train,  (AxdynF64, Ix2) };

            /*
               For kmeans, we will set the target to Array2's default with respect to the records length.
               This is because the KMeans algorithm doesn't rely on the targets.

               But in order for the algorithm to work correctly and keep a unified for `prepare_train_data`, we process all the targets same
               and then later set the targets for kmeans to defaults.
            */

            let records_shape = train.records().shape().to_vec();

            let train = train
                .with_targets::<Array2<f64>>(Array2::zeros((records_shape[0], records_shape[1])));
            let model = kmeans(
                n_runs.into(),
                n_clusters.into(),
                tolerance,
                max_n_iterations,
                init_method,
                random_state,
            );
            Ok(SupportedModels::KMeans(to_status_error(model.fit(&train))?))
        }
        Models::LinearRegression { fit_intercept } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::Float64)?));

            let train = prepare_train_data! {"LinearRegression", train,  (AxdynF64, Ix1) };

            let model = linear_regression(fit_intercept);

            Ok(SupportedModels::LinearRegression(to_status_error(
                model.fit(&train),
            )?))
        }
        Models::TweedieRegressor {
            fit_intercept,
            alpha,
            max_iter,
            link,
            tol,
            power,
        } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::Float64)?));

            let train = prepare_train_data! {"TweedieRegressor", train,  (AxdynF64, Ix1) };

            let model = tweedie_regression(fit_intercept, alpha, max_iter, link, tol, power);

            Ok(SupportedModels::TweedieRegressor(to_status_error(
                model.fit(&train),
            )?))
        }

        Models::BinomialLogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
            shape,
        } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::UInt64)?));

            let train =
                prepare_train_data! {"BinomialLogisticRegression", train,  (AxdynU64, Ix1) };

            let model = binomial_logistic_regression(
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
                shape,
            )?;
            Ok(SupportedModels::BinomialLogisticRegression(
                to_status_error(model.fit(&train))?,
            ))
        }
        Models::MultinomialLogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
            shape,
        } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::UInt64)?));

            let train =
                prepare_train_data! {"MultinomialLogisticRegression", train,  (AxdynU64, Ix1) };

            let model = multinomial_logistic_regression(
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
                shape,
            )?;
            Ok(SupportedModels::MultinomialLogisticRegression(
                to_status_error(model.fit(&train))?,
            ))
        }

        Models::DecisionTree {
            split_quality,
            max_depth,
            min_weight_split,
            min_weight_leaf,
            min_impurity_decrease,
        } => {
            let train = train.with_targets(IArrayStore(targets.cast(ArrayStoreType::UInt64)?));

            let train = prepare_train_data! {"DecisionTree", train,  (AxdynU64, Ix1) };

            let train = train.map_targets(|t| LabelU64(*t));
            let model = decision_trees(
                split_quality,
                max_depth,
                min_weight_split,
                min_weight_leaf,
                min_impurity_decrease,
            );
            Ok(SupportedModels::DecisionTree(to_status_error(
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
            todo!()
        }
    }
}

/// This method is used to run a prediction on an already fitted model, based on the model selection type.
pub fn predict(
    model: Arc<SupportedModels>,
    data: ArrayStore,
    probability: bool,
) -> Result<ArrayStore, Status> {
    let sample = IArrayStore(data.cast(ArrayStoreType::Float64)?);
    let sample = get_inner_array! {AxdynF64, sample, Ix2, "Ix2", "predict", "sample"};
    let prediction = match &*model {
        SupportedModels::ElasticNet(m) => {
            let results = m.predict(sample).targets.into_dyn();
            ArrayStore::AxdynF64(results)
        }
        SupportedModels::GaussianNaiveBayes(m) => {
            let results = m.predict(sample).map_targets(|t| t.0).targets.into_dyn();
            ArrayStore::AxdynU64(results)
        }
        SupportedModels::KMeans(m) => {
            let results = m
                .predict(sample)
                .map_targets(|t| *t as u64)
                .targets
                .into_dyn();
            ArrayStore::AxdynU64(results)
        }
        SupportedModels::LinearRegression(m) => {
            let results = m.predict(sample).targets.into_dyn();
            ArrayStore::AxdynF64(results)
        }
        SupportedModels::BinomialLogisticRegression(m) => {
            let result = if probability {
                let result = m.predict_probabilities(&sample).into_dyn();
                ArrayStore::AxdynF64(result)
            } else {
                let result = m.predict(sample).targets.into_dyn();
                ArrayStore::AxdynU64(result)
            };

            result
        }
        SupportedModels::MultinomialLogisticRegression(m) => {
            let result = if probability {
                let result = m.predict_probabilities(&sample).into_dyn();
                ArrayStore::AxdynF64(result)
            } else {
                let result = m.predict(sample).targets.into_dyn();
                ArrayStore::AxdynU64(result)
            };

            result
        }
        SupportedModels::DecisionTree(m) => {
            let results = m.predict(sample).map_targets(|t| t.0).targets.into_dyn();
            ArrayStore::AxdynU64(results)
        }
        _ => return Err(Status::failed_precondition("Unsupported Model")),
    };

    Ok(prediction)
}

#[allow(unused)]
pub fn inner_cross_validate(
    model: Models,
    records: ArrayStore,
    targets: ArrayStore,
    scoring: &Scoring,
    cv: usize,
) -> Result<ArrayStore, Status> {
    let mut train = get_datasets(records, targets);
    let get_score = |scoring, prediction, truth| match scoring {
        Scoring::RegressionMetric(metric) => metric
            .regression_metric
            .clone()
            .unwrap()
            .compute(prediction, truth),
        Scoring::ClassificationMetric(metric) => metric
            .classification_metric
            .clone()
            .unwrap()
            .compute(prediction, truth),
    };
    let result = match model {
        Models::LinearRegression { fit_intercept } => {
            let m = linear_regression(fit_intercept);
            let mut train = prepare_train_data! {"LinearRegression", train,  (AxdynF64, Ix1) };
            todo!()
        }

        Models::BinomialLogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
            shape,
        } => {
            let m = binomial_logistic_regression(
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
                shape,
            )?;

            todo!()
            // let mut train = prepare_train_data! {"LogisticRegression", train,  (AxdynU64, Ix1) };

            // let mut train = train.map_targets(|t| LabelU64(*t));

            // let arr = to_status_error(train.cross_validate_single(
            //     cv,
            //     &vec![m][..],
            //     |pred, truth| classification_metrics(pred, truth, scoring),
            // ))?;

            // ArrayStore::AxdynF32(arr.into_dyn())
        }

        Models::MultinomialLogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
            shape,
        } => {
            let m = multinomial_logistic_regression(
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
                shape,
            )?;
            todo!()
        }

        _ => {
            return Err(Status::failed_precondition(format!(
                "Unsupported Model: {:?}",
                model
            )))
        }
    };

    Ok(result)
}
