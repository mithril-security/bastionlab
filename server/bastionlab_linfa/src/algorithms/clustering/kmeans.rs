use linfa::{traits::Fit, DatasetBase};
use linfa_clustering::{GmmError, KMeans, KMeansInit};
use linfa_nn::distance::L2Dist;
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr};

pub fn kmeans(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>,
    n_runs: usize,
    n_clusters: usize,
    tolerance: f64,
    max_n_iterations: u64,
    init_method: KMeansInit<f64>,
) -> Result<KMeans<f64, L2Dist>, GmmError> {
    let model = KMeans::params(n_clusters)
        .n_runs(n_runs)
        .max_n_iterations(max_n_iterations)
        .tolerance(tolerance)
        .init_method(init_method)
        .fit(&train)?;

    Ok(model)
}
