use tch::TchError;

/// Type for the state of the [`Optimizer`].
///
/// **NB**: This enum has to be updated once more optimizers are added.
#[derive(Debug)]
pub enum OptimizerStateType {
    SGD {
        statistics: Vec<u8>,
    },
    Adam {
        m: Vec<u8>,
        v: Vec<u8>,
        v_hat_max: Vec<u8>,
        t: i32,
    },
}

/// Common interface for all optimizers
pub trait Optimizer {
    /// Sets the accumulated gradients of all trained parameters to zero.
    fn zero_grad(&mut self) -> Result<(), TchError>;
    /// Performs a single training step using the accumulated gradients.
    fn step(&mut self) -> Result<(), TchError>;
    /// Returns contained parameters as [`Vec<u8>`].
    fn into_bytes(&mut self) -> Result<Vec<u8>, TchError>;
    /// Saves the latest state of the [`Optimizer`] as [`OptimizerStateType`]
    fn get_state(&mut self) -> Result<OptimizerStateType, TchError>;
}
