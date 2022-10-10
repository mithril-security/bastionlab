use std::sync::Mutex;

use tch::Tensor;

mod adam;
mod optimizer;
mod sgd;

fn initialize_statistics(length: usize) -> Vec<Option<Tensor>> {
    let mut v = Vec::with_capacity(length);
    for _ in 0..length {
        v.push(None);
    }
    v
}

pub fn log_checkpoint(inner_params: Vec<(String, Tensor)>) -> CheckPoint {
    inner_params
        .iter()
        .map(|(n, v)| (n.clone(), Mutex::new(v.copy().f_detach_().unwrap())))
        .collect()
}

pub use adam::Adam;
pub use optimizer::Optimizer;
pub use sgd::SGD;

use crate::nn::CheckPoint;
