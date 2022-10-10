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

pub use adam::Adam;
pub use optimizer::Optimizer;
pub use sgd::SGD;
