use std::error::Error;

use linfa::{DatasetBase, PlattParams};
use linfa_bayes::GaussianNb;
use linfa_clustering::{KMeans, KMeansInit};
use linfa_elasticnet::ElasticNet;
use linfa_kernel::{Kernel, KernelMethod, KernelParams, KernelType};
use linfa_linear::FittedLinearRegression;
use linfa_logistic::FittedLogisticRegression;
use linfa_nn::{
    distance::L2Dist, BallTreeIndex, CommonNearestNeighbour, KdTreeIndex, LinearSearchIndex,
};
use linfa_svm::Svm;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array2, ArrayBase, Ix1, Ix2, OwnedRepr};
use polars::{
    error::ErrString,
    prelude::{PolarsError, PolarsResult},
};
use tonic::{Request, Status};

use crate::{
    linfa_proto::{
        ElasticNet as BastionElasticNet, KMeans as BastionKMeans, Svm as BastionSvm, Trainer, *,
    },
    to_status_error,
};

pub enum Models {
    GaussianNaiveBayes {
        var_smoothing: f64,
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

    LogisticRegression {
        alpha: f64,
        gradient_tolerance: f64,
        fit_intercept: bool,
        max_iterations: u64,
        initial_params: Option<Vec<f64>>,
    },

    DecisionTree {
        split_quality: SplitQuality,
        max_depth: Option<usize>,
        min_weight_split: f32,
        min_weight_leaf: f32,
        min_impurity_decrease: f64,
    },
    SVM {
        c: Vec<f32>,
        eps: Option<f32>,
        nu: Option<f32>,
        shrinking: bool,
        platt_params: PlattParams<f64, ()>,
        kernel_params: KernelParams<f64>,
    },
}

#[allow(unused)]
#[derive(Debug)]
pub enum SupportedModels {
    GaussianNaiveBayes(GaussianNb<f64, usize>),
    ElasticNet(ElasticNet<f64>),
    KMeans(KMeans<f64, L2Dist>),
    LinearRegression(FittedLinearRegression<f64>),
    LogisticRegression(FittedLogisticRegression<f64, usize>),
    DecisionTree(DecisionTree<f64, usize>),
    BallTree(BallTreeIndex<'static, f64, L2Dist>),
    Linear(LinearSearchIndex<'static, f64, L2Dist>),
    KdTree(KdTreeIndex<'static, f64, L2Dist>),
    SVM(Svm<f64, f64>),
}

#[derive(Debug)]
pub enum PredictionTypes {
    Usize(DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>),
    Float(DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>),
    Probability(ArrayBase<OwnedRepr<f64>, Ix1>),
}

pub fn to_polars_error<T, E: Error>(input: Result<T, E>) -> PolarsResult<T> {
    input.map_err(|err| PolarsError::InvalidOperation(ErrString::Owned(err.to_string())))
}

pub fn get_datasets<D: Clone, T>(
    records: ArrayBase<OwnedRepr<D>, Ix2>,
    target: ArrayBase<OwnedRepr<T>, Ix1>,
    col_names: Vec<String>,
) -> PolarsResult<DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, Ix1>>> {
    // ** For now, shuffling is not implemented.
    let dataset = linfa::Dataset::new(records, target);
    let dataset = dataset.with_feature_names::<String>(col_names);
    Ok(dataset)
}

pub fn process_trainer_req(
    request: Request<TrainingRequest>,
) -> Result<(String, String, Option<Trainer>), Status> {
    Ok((
        request.get_ref().records.clone(),
        request.get_ref().target.clone(),
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
    if let Some(t) = trainer.gaussian_nb {
        Ok(Models::GaussianNaiveBayes {
            var_smoothing: t.var_smoothing.into(),
        })
    } else if let Some(t) = trainer.linear_regression {
        Ok(Models::LinearRegression {
            fit_intercept: t.fit_intercept,
        })
    } else if let Some(t) = trainer.logistic_regression {
        let initial_params = t
            .initial_params
            .iter()
            .map(|v| (*v).into())
            .collect::<Vec<f64>>();

        let initial_params = if initial_params.len() == 0 {
            None
        } else {
            Some(initial_params)
        };
        Ok(Models::LogisticRegression {
            alpha: t.alpha.into(),
            gradient_tolerance: t.gradient_tolerance.into(),
            fit_intercept: t.fit_intercept,
            max_iterations: t.max_iterations,
            initial_params,
        })
    } else if let Some(t) = trainer.decision_tree {
        {
            let split_quality = match t.split_quality {
                Some(v) => match v {
                    decision_tree::SplitQuality::Gini(decision_tree::Gini {}) => SplitQuality::Gini,
                    decision_tree::SplitQuality::Entropy(decision_tree::Entropy {}) => {
                        SplitQuality::Entropy
                    }
                },
                None => {
                    return Err(Status::failed_precondition("SplitQuality not found!"));
                }
            };

            let max_depth: Option<usize> = match t.max_depth {
                Some(v) => Some(v as usize),
                None => {
                    return Err(Status::failed_precondition("max_depth not provided!"));
                }
            };

            let min_impurity_decrease = t.min_impurity_decrease.into();

            Ok(Models::DecisionTree {
                split_quality,
                max_depth,
                min_weight_split: t.min_weight_split,
                min_weight_leaf: t.min_weight_leaf,
                min_impurity_decrease,
            })
        }
    } else if let Some(t) = trainer.elastic_net {
        let BastionElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        } = t;
        Ok(Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        })
    } else if let Some(t) = trainer.kmeans {
        let BastionKMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
        } = t;
        let init_method =
            init_method.ok_or_else(|| Status::aborted("Invalid KMeans Init Method!"))?;
        let init_method: KMeansInit<f64> = match init_method {
            k_means::InitMethod::Random(k_means::Random {}) => KMeansInit::Random,
            k_means::InitMethod::PreComputed(k_means::Precomputed {
                list,
                n_centroids,
                n_features,
            }) => {
                let sh = (n_centroids as usize, n_features as usize);
                let list = to_status_error(Array2::from_shape_vec(
                    sh,
                    list.into_iter().map(|v| v.into()).collect::<Vec<_>>(),
                ))?;

                KMeansInit::Precomputed(list)
            }
            k_means::InitMethod::KmeansPara(k_means::KMeansPara {}) => KMeansInit::KMeansPara,
            k_means::InitMethod::KmeansPlusPlus(k_means::KMeansPlusPlus {}) => {
                KMeansInit::KMeansPlusPlus
            }
        };
        Ok(Models::KMeans {
            n_runs: n_runs.try_into().unwrap(),
            n_clusters: n_clusters.try_into().unwrap(),
            tolerance: tolerance.into(),
            max_n_iterations,
            init_method,
        })
    } else if let Some(t) = trainer.svm {
        let BastionSvm {
            c,
            eps,
            nu,
            shrinking,
            platt_params,
            kernel_params,
        } = t;

        let platt_params: PlattParams<f64, ()> = match platt_params {
            Some(v) => PlattParams::default()
                .maxiter(v.maxiter as usize)
                .sigma(v.sigma as f64)
                .minstep(v.ministep as f64),
            None => {
                return Err(Status::aborted(
                    "Could not deserialize PlattParams in Svm".to_string(),
                ));
            }
        };

        let kernel_params = match &kernel_params {
            Some(v) => {
                let kernel_type = match &v.kernel_type {
                    Some(t) => match t {
                        svm::kernel_params::KernelType::Dense(_) => KernelType::Dense,
                        svm::kernel_params::KernelType::Sparse(s) => {
                            KernelType::Sparse(s.sparsity as usize)
                        }
                    },
                    None => {
                        return Err(Status::aborted(
                            "Could not deserialize KernelType in KernelParams in Svm".to_string(),
                        ));
                    }
                };

                // Verify this.
                let n = match v.n() {
                    svm::kernel_params::N::LinearSearch => CommonNearestNeighbour::LinearSearch,
                    svm::kernel_params::N::KdTree => CommonNearestNeighbour::KdTree,
                    svm::kernel_params::N::BallTree => CommonNearestNeighbour::BallTree,
                };

                let kernel_method = match &v.kernel_method {
                    Some(m) => match m {
                        svm::kernel_params::KernelMethod::Guassian(g) => {
                            KernelMethod::Gaussian(g.eps as f64)
                        }
                        svm::kernel_params::KernelMethod::Linear(_) => KernelMethod::Linear,
                        svm::kernel_params::KernelMethod::Poly(p) => {
                            KernelMethod::Polynomial(p.constant as f64, p.degree as f64)
                        }
                    },
                    None => {
                        return Err(Status::aborted(
                            "Could not deserialize KernelMethod in KernelParams in Svm".to_string(),
                        ));
                    }
                };

                Kernel::params()
                    .kind(kernel_type)
                    .method(kernel_method)
                    .nn_algo(n)
            }
            None => {
                return Err(Status::aborted(
                    "Could not deserialize KernelParams in Svm".to_string(),
                ));
            }
        };
        Ok(Models::SVM {
            c,
            eps,
            nu,
            shrinking,
            platt_params,
            kernel_params,
        })
    } else {
        return Err(Status::not_found(format!("Unknown trainer: {:?}", trainer)));
    }
}
