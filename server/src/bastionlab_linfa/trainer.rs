use std::error::Error;

use linfa::DatasetBase;
use linfa_bayes::GaussianNb;
use linfa_clustering::{KMeans, KMeansInit};
use linfa_elasticnet::ElasticNet;
use linfa_linear::FittedLinearRegression;
use linfa_nn::distance::L2Dist;
use ndarray::{Array2, ArrayBase, Ix1, Ix2, OwnedRepr};
use polars::{
    error::ErrString,
    prelude::{PolarsError, PolarsResult},
};
use tonic::{Request, Status};

use crate::{
    bastionlab_linfa::to_status_error,
    grpc::{
        training_request::{self, Trainer},
        TrainingRequest,
    },
    to_type,
};

pub enum Models {
    GaussianNaiveBayes {
        var_smoothing: f32,
    },
    ElasticNet {
        penalty: f32,
        l1_ratio: f32,
        with_intercept: bool,
        max_iterations: u32,
        tolerance: f32,
    },
    KMeans {
        n_runs: usize,
        n_clusters: usize,
        tolerance: f64,
        max_n_iterations: u64,
        init_method: KMeansInit<f64>,
    },
    LinearRegression {
        fit_intercept: bool,
    },
}

#[derive(Debug, Clone)]
pub enum SupportedModels {
    GaussianNaiveBayes(GaussianNb<f64, usize>),
    ElasticNet(ElasticNet<f64>),
    KMeans(KMeans<f64, L2Dist>),
    LinearRegression(FittedLinearRegression<f64>),
}

#[derive(Debug)]
pub enum PredictionTypes {
    Usize(DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>),
    Float(DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>),
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

pub fn process_trainer_req(
    request: Request<TrainingRequest>,
) -> Result<(String, String, f32, Option<Trainer>), Status> {
    Ok((
        request.get_ref().records.clone(),
        request.get_ref().target.clone(),
        request.get_ref().ratio,
        Some(
            request
                .get_ref()
                .trainer
                .clone()
                .ok_or_else(|| Status::aborted("Trainer not supported!"))?,
        ),
    ))
}

pub fn select_trainer(trainer: Trainer) -> Result<Models, Status> {
    match trainer {
        Trainer::GaussianNb(training_request::GaussianNb { var_smoothing }) => {
            Ok(Models::GaussianNaiveBayes { var_smoothing })
        }
        Trainer::LinearRegression(training_request::LinearRegression { fit_intercept }) => {
            Ok(Models::LinearRegression { fit_intercept })
        }
        Trainer::ElasticNet(training_request::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        }) => Ok(Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        }),
        Trainer::Kmeans(training_request::KMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
        }) => {
            let init_method =
                init_method.ok_or_else(|| Status::aborted("Invalid KMeans Init Method!"))?;
            let init_method: KMeansInit<f64> = match init_method {
                training_request::k_means::InitMethod::Random(
                    training_request::k_means::Random {},
                ) => KMeansInit::Random,
                training_request::k_means::InitMethod::PreComputed(
                    training_request::k_means::Precomputed {
                        list,
                        n_centroids,
                        n_features,
                    },
                ) => {
                    let list = to_type! {<f64>(list)};
                    let sh = (n_centroids as usize, n_features as usize);
                    let list = to_status_error(Array2::from_shape_vec(sh, list))?;
                    KMeansInit::Precomputed(list)
                }
                training_request::k_means::InitMethod::KmeansPara(
                    training_request::k_means::KMeansPara {},
                ) => KMeansInit::KMeansPara,
                training_request::k_means::InitMethod::KmeansPlusPlus(
                    training_request::k_means::KMeansPlusPlus {},
                ) => KMeansInit::KMeansPlusPlus,
            };
            Ok(Models::KMeans {
                n_runs: n_runs.try_into().unwrap(),
                n_clusters: n_clusters.try_into().unwrap(),
                tolerance: tolerance.into(),
                max_n_iterations,
                init_method,
            })
        }
    }
}
