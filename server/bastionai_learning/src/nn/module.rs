use tch::Tensor;
use tch::{TrainableCModule, nn::VarStore, TchError, Device};
use std::sync::{Arc, RwLock};
use super::{Parameters, LossType};
use crate::procedures::{ModuleTrainer, ModuleTester, Metric};
use crate::data::Dataset;
use crate::optim::Optimizer;
use crate::serialization::SizedObjectsBytes;

pub enum Privacy {
    Standard,
    DifferentialPrivacy {
        max_grad_norm: f64,
        noise_multiplier: f64,
    }
}

/// A Trainable Model
///
/// Contains a reference to a `tch::TrainableCModule`
/// obtained from bytes data that has been serialized with
/// `torch.jit.save` and a VarStore that holds the models
/// parameters.
#[derive(Debug)]
pub struct Module {
    c_module: TrainableCModule,
    var_store: VarStore,
}

impl Module {
    pub fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, TchError> {
        self.c_module.forward_ts(inputs)
    }
    /// Get the model's parameters wrapped in a `Parameter::Standard` variant
    /// to train the model without differential privacy with an [`Optimizer`].
    pub fn parameters(&self) -> Parameters {
        Parameters::standard(&self.var_store)
    }
    /// Get the model's parameters wrapped in a `Parameter::Private` variant
    /// to train the model with DP-SGD with an [`Optimizer`].
    pub fn private_parameters(
        &self,
        max_grad_norm: f64,
        noise_multiplier: f64,
        loss_type: LossType,
    ) -> Parameters {
        Parameters::private(&self.var_store, max_grad_norm, noise_multiplier, loss_type)
    }
    /// Moves all the parameters to the specified device.
    pub fn set_device(&mut self, device: Device) {
        self.var_store.set_device(device);
    }
    /// Returns an ietrator that trains the model using a very basic training loop on specified `device`
    /// and that yields the loss after every iteration (i.e. every batch).
    /// Loss, optimizer, batch size and more are read from the given `config`.
    pub fn train(
        s: Arc<RwLock<Self>>,
        dataset: Arc<RwLock<Dataset>>,
        optimizer: Box<dyn Optimizer + Send>,
        metric: Metric,
        epochs: usize,
        batch_size: usize,
        device: Device,
    ) -> ModuleTrainer {
        let mut module = s.write().unwrap();
        module.set_device(device);

        ModuleTrainer::new(
            Arc::clone(&s),
            dataset,
            optimizer,
            metric,
            device,
            epochs,
            batch_size,
        )
    }
    /// Tests the model using a very basic test loop on specified `device`.
    /// Metric, batch size and more are read from the given `config`.
    pub fn test(
        s: Arc<RwLock<Module>>,
        dataset: Arc<RwLock<Dataset>>,
        metric: Metric,
        batch_size: usize,
        device: Device,
    ) -> ModuleTester {
        s.write().unwrap().set_device(device);

        ModuleTester::new(s, dataset, metric, device, batch_size)
    }
}

impl TryFrom<SizedObjectsBytes> for Module {
    type Error = TchError;

    fn try_from(mut value: SizedObjectsBytes) -> Result<Self, Self::Error> {
        let object = value.next().ok_or(TchError::FileFormat(String::from(
            "Invalid data, expected at least one object in stream.",
        )))?;
        let vs = VarStore::new(Device::Cpu);
        Ok(Module {
            c_module: TrainableCModule::load_data(&mut &object[..], vs.root())?,
            var_store: vs,
        })
    }
}

impl TryFrom<&Module> for SizedObjectsBytes {
    type Error = TchError;

    fn try_from(value: &Module) -> Result<Self, Self::Error> {
        let mut module_bytes = SizedObjectsBytes::new();
        let mut buf = Vec::new();
        value.var_store.save_to_stream(&mut buf)?;
        module_bytes.append_back(buf);
        Ok(module_bytes)
    }
}
