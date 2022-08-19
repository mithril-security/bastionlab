use tch::Tensor;

mod optimizer;
mod sgd;
mod adam;

fn initialize_statistics(length: usize) -> Vec<Option<Tensor>> {
    let mut v = Vec::with_capacity(length);
    for _ in 0..length {
        v.push(None);
    }
    v
}

pub use optimizer::Optimizer;
pub use sgd::SGD;
pub use adam::Adam;
