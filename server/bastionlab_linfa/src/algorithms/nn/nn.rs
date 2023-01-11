use linfa_nn::{distance::Distance, *};
use ndarray::{ArrayBase, Data, Ix2};

pub fn linear_search<DT: Data<Elem = f64>, D: Distance<f64>>(
    batch: &ArrayBase<DT, Ix2>,
    distance: D,
) -> Result<LinearSearchIndex<f64, D>, BuildError> {
    LinearSearchIndex::new(batch, distance)
}

pub fn kdtree<DT: Data<Elem = f64>, D: Distance<f64>>(
    batch: &ArrayBase<DT, Ix2>,
    leaf_size: usize,
    dist_fn: D,
) -> Result<KdTreeIndex<f64, D>, BuildError> {
    KdTreeIndex::new(batch, leaf_size, dist_fn)
}

pub fn balltree<DT: Data<Elem = f64>, D: Distance<f64>>(
    batch: &ArrayBase<DT, Ix2>,
    leaf_size: usize,
    dist_fn: D,
) -> Result<BallTreeIndex<f64, D>, BuildError> {
    BallTreeIndex::new(batch, leaf_size, dist_fn)
}
