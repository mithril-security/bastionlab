use tch::{COptimizer, TchError};

/// Common interface for all optimizers
pub trait Optimizer {
    /// Sets the accumulated gradients of all trained parameters to zero.
    fn zero_grad(&mut self) -> Result<(), TchError>;
    /// Performs a single training step using the accumulated gradients.
    fn step(&mut self) -> Result<(), TchError>;
    /// Checkpoints model during training
    fn check_point(&mut self) -> Result<(), TchError>;
}

impl Optimizer for COptimizer {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        COptimizer::zero_grad(self)
    }
    fn step(&mut self) -> Result<(), TchError> {
        COptimizer::step(self)
    }

    fn check_point(&mut self) -> Result<(), TchError> {
        todo!()
    }
}
