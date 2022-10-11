use tch::{COptimizer, TchError};

/// Type for the state of the [`Optimizer`].
///
/// **NB**: This enum has to updated once more optimizers are added.
pub enum OptimizerStateType {
    SGD(Vec<u8>),
    Adam(Vec<u8>, Vec<u8>, Vec<u8>, i32),
}

/// Common interface for all optimizers
pub trait Optimizer {
    /// Sets the accumulated gradients of all trained parameters to zero.
    fn zero_grad(&mut self) -> Result<(), TchError>;
    /// Performs a single training step using the accumulated gradients.
    fn step(&mut self) -> Result<(), TchError>;
    /// Returns contained parameters as [`Vec<u8>`].
    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError>;
    /// Saves the current state of the [`Optimizer`] as [`OptimizerStateType`]
    fn get_state(&mut self) -> Result<OptimizerStateType, TchError>;
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
    fn get_state(&mut self) -> Result<OptimizerStateType, TchError> {
        todo!()
    }
}
