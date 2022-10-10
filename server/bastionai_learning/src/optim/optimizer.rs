use tch::{COptimizer, TchError};

/// Common interface for all optimizers
pub trait Optimizer {
    /// Sets the accumulated gradients of all trained parameters to zero.
    fn zero_grad(&mut self) -> Result<(), TchError>;
    /// Performs a single training step using the accumulated gradients.
    fn step(&mut self) -> Result<(), TchError>;
    /// Returns contained parameters as [`Vec<u8>`].
    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError>;
}

impl Optimizer for COptimizer {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        COptimizer::zero_grad(self)
    }
    fn step(&mut self) -> Result<(), TchError> {
        COptimizer::step(self)
    }
    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError> {
        todo!()
    }
}
