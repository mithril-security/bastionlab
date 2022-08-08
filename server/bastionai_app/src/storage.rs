use crate::remote_torch::{TrainConfig, TestConfig, train_config};
use crate::utils::*;
use ring::hmac;
use std::convert::{TryFrom, TryInto};
use std::io::Cursor;
use std::sync::Mutex;
use tch::nn::VarStore;
use tch::{Device, TchError, Tensor, TrainableCModule, IndexOp};
use private_learning::{Parameters, LossType, SGD, Optimizer, l2_loss, Adam};
use std::sync::{Arc, RwLock};
use rand::{seq::SliceRandom, thread_rng};

/// A buffer to serialize/deserialize byte data in the following format:
/// 
/// `obj = [length: 8 bytes, little-endian | data: length bytes]`
/// 
/// `stream = [obj, ...]`
#[derive(Debug)]
pub struct SizedObjectsBytes(Vec<u8>);

impl SizedObjectsBytes {
    /// returns a new empty `SizedObjectBytes` buffer.
    pub fn new() -> Self {
        SizedObjectsBytes(Vec::new())
    }

    /// Adds a new bytes object to the buffer, its size is automatically
    /// converted to little-endian bytes and prepended to the data.
    pub fn append_back(&mut self, mut bytes: Vec<u8>) {
        self.0.extend_from_slice(&bytes.len().to_le_bytes());
        self.0.append(&mut bytes);
    }

    /// Removes the eight first bytes of the buffer and interprets them
    /// as the little-endian bytes of the length. Then removes and returns
    /// the next length bytes from the buffer.
    pub fn remove_front(&mut self) -> Vec<u8> {
        let len = read_le_usize(&mut &self.0.drain(..8).collect::<Vec<u8>>()[..]);
        self.0.drain(..len).collect()
    }

    /// Get raw bytes.
    pub fn get(&self) -> &Vec<u8> {
        &self.0
    }
}

impl From<SizedObjectsBytes> for Vec<u8> {
    fn from(value: SizedObjectsBytes) -> Self {
        value.0
    }
}

impl From<Vec<u8>> for SizedObjectsBytes {
    fn from(value: Vec<u8>) -> Self {
        SizedObjectsBytes(value)
    }
}

impl Iterator for SizedObjectsBytes {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.len() > 0 {
            Some(self.remove_front())
        } else {
            None
        }
    }
}

pub struct ModuleTrainer {
    module: Arc<RwLock<Module>>,
    dataset: Arc<RwLock<Dataset>>,
    optimizer: Box<dyn Optimizer + Send>,
    metric: Metric,
    device: Device,
    epochs: usize,
    batch_size: usize,
    dataloader: std::iter::Enumerate<DatasetIter>,
    current_epoch: usize,
}

impl ModuleTrainer {
    pub fn new(module: Arc<RwLock<Module>>, dataset: Arc<RwLock<Dataset>>, optimizer: Box<dyn Optimizer + Send>, metric: Metric, device: Device, epochs: usize, batch_size: usize) -> ModuleTrainer {
        ModuleTrainer {
            module,
            dataset: Arc::clone(&dataset),
            optimizer,
            metric,
            device,
            epochs,
            batch_size,
            dataloader: Dataset::iter_shuffle(dataset, batch_size).enumerate(),
            current_epoch: 0,
        }
    }

    pub fn train_on_batch(&mut self, i: usize, input: Tensor, label: Tensor) -> Result<(i32, i32, f32), TchError> {
        let input = input.f_to(self.device)?;
        let label = label.f_to(self.device)?;
        let output = self.module.read().unwrap().c_module.forward_ts(&[input])?;
        let loss = self.metric.compute(&output, &label)?;
        self.optimizer.zero_grad()?;
        loss.backward();
        self.optimizer.step()?;
        Ok((self.current_epoch as i32, i as i32, self.metric.value()))
    }

    pub fn nb_epochs(&self) -> usize {
        self.epochs
    }

    pub fn nb_batches(&self) -> usize {
        self.dataset.read().unwrap().len() / self.batch_size
    }
}

impl Iterator for ModuleTrainer {
    type Item = Result<(i32, i32, f32), TchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((i, (input, label))) = self.dataloader.next() {
            Some(self.train_on_batch(i, input, label))
        } else {
            self.current_epoch += 1;
            self.metric.reset();
            if self.current_epoch < self.epochs {
                self.dataloader = Dataset::iter_shuffle(Arc::clone(&self.dataset), self.batch_size).enumerate();
                self.next()
            } else {
                None
            }
        }
    }
}

pub struct ModuleTester {
    module: Arc<RwLock<Module>>,
    metric: Metric,
    device: Device,
    dataloader: std::iter::Enumerate<DatasetIter>,
    nb_batches: usize,
}

impl ModuleTester {
    pub fn new(module: Arc<RwLock<Module>>, dataset: Arc<RwLock<Dataset>>, metric: Metric, device: Device, batch_size: usize) -> ModuleTester {
        let nb_batches = dataset.read().unwrap().len() / batch_size;
        ModuleTester {
            module,
            metric,
            device,
            dataloader: Dataset::iter_shuffle(dataset, batch_size).enumerate(),
            nb_batches,
        }
    }

    pub fn test_on_batch(&mut self, i: usize, input: Tensor, label: Tensor) -> Result<(i32, f32), TchError> {
        let input = input.f_to(self.device)?;
        let label = label.f_to(self.device)?;
        let output = self.module.read().unwrap().c_module.forward_ts(&[input])?;
        let _ = self.metric.compute(&output, &label)?;
        Ok((i as i32, self.metric.value()))
    }

    pub fn nb_batches(&self) -> usize {
        self.nb_batches
    }
}

impl Iterator for ModuleTester {
    type Item = Result<(i32, f32), TchError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((i, (input, label))) = self.dataloader.next() {
            Some(self.test_on_batch(i, input, label))
        } else {
            None
        }
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
    /// Get the model's parameters wrapped in a `Parameter::Standard` variant
    /// to train the model without differential privacy with an [`Optimizer`].
    pub fn parameters(&self) -> Parameters {
        Parameters::standard(&self.var_store)
    }
    /// Get the model's parameters wrapped in a `Parameter::Private` variant
    /// to train the model with DP-SGD with an [`Optimizer`].
    pub fn private_parameters(&self, max_grad_norm: f64, noise_multiplier: f64, loss_type: LossType) -> Parameters {
        Parameters::private(&self.var_store, max_grad_norm, noise_multiplier, loss_type)
    }
    /// Moves all the parameters to the specified device.
    pub fn set_device(&mut self, device: Device) {
        self.var_store.set_device(device);
    }
    /// Returns an ietrator that trains the model using a very basic training loop on specified `device`
    /// and that yields the loss after every iteration (i.e. every batch).
    /// Loss, optimizer, batch size and more are read from the given `config`.
    pub fn train(s: Arc<RwLock<Self>>, dataset: Arc<RwLock<Dataset>>, config: TrainConfig, device: Device) -> Result<ModuleTrainer, TchError> {
        let mut module = s.write().unwrap();
        module.set_device(device);

        let parameters = match config.privacy.ok_or(TchError::FileFormat(String::from("Invalid privacy option")))? {
            train_config::Privacy::Standard(_) => module.parameters(),
            train_config::Privacy::DifferentialPrivacy(train_config::DpParameters { max_grad_norm, noise_multiplier }) => 
                module.private_parameters(max_grad_norm as f64, noise_multiplier as f64, private_learning::LossType::Mean(config.batch_size as i64)),
        };
        
        let optimizer = match config.optimizer.ok_or(TchError::FileFormat(String::from("Invalid optimizer")))? {
            train_config::Optimizer::Sgd(train_config::Sgd { learning_rate, weight_decay, momentum, dampening, nesterov }) =>
                Box::new(SGD::new(parameters, learning_rate as f64)
                    .weight_decay(weight_decay as f64)
                    .momentum(momentum as f64)
                    .dampening(dampening as f64)
                    .nesterov(nesterov)) as Box<dyn Optimizer + Send>,
            train_config::Optimizer::Adam(train_config::Adam { learning_rate, beta_1, beta_2, epsilon, weight_decay, amsgrad }) =>
                Box::new(Adam::new(parameters, learning_rate as f64)
                    .beta_1(beta_1 as f64)
                    .beta_2(beta_2 as f64)
                    .epsilon(epsilon as f64)
                    .weight_decay(weight_decay as f64)
                    .amsgrad(amsgrad)) as Box<dyn Optimizer + Send>,
        };
        
        let metric = Metric::try_from_name(&config.metric)?;
        
        Ok(ModuleTrainer::new(Arc::clone(&s), dataset, optimizer, metric, device, config.epochs as usize, config.batch_size as usize))
    }
    /// Tests the model using a very basic test loop on specified `device`.
    /// Metric, batch size and more are read from the given `config`.
    pub fn test(s: Arc<RwLock<Module>>, dataset: Arc<RwLock<Dataset>>, config: TestConfig, device: Device) -> Result<ModuleTester, TchError> {
        s.write().unwrap().set_device(device);

        let mut metric = Metric::try_from_name(&config.metric)?;
        Ok(ModuleTester::new(s, dataset, metric, device, config.batch_size as usize))
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

/// Simple in-memory dataset
#[derive(Debug)]
pub struct Dataset {
    samples: Mutex<Tensor>,
    labels: Mutex<Tensor>,
}

/// Simple iterator over [`Dataset`].
pub struct DatasetIter {
    dataset: Arc<RwLock<Dataset>>,
    indexes: Vec<i64>,
    batch_size: usize,
}

impl Iterator for DatasetIter {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.indexes.len() < self.batch_size {
            None
        } else {
            let indexes = self.indexes.drain(..self.batch_size);
            let dataset = self.dataset.read().unwrap();
            let samples = dataset.samples.lock().unwrap();
            let labels = dataset.labels.lock().unwrap();
            let items = indexes.map(|idx| (samples.i(idx), labels.i(idx)));
            let mut samples = Vec::with_capacity(self.batch_size);
            let mut labels = Vec::with_capacity(self.batch_size);
            for (sample, label) in items {
                samples.push(sample);
                labels.push(label);
            }
            let samples = Tensor::stack(&samples, 0);
            let labels = Tensor::stack(&labels, 0);
            Some((samples, labels))
        }
    }
}

impl Dataset {
    /// Returns an iterator over this dataset.
    pub fn iter_shuffle(s: Arc<RwLock<Self>>, batch_size: usize) -> DatasetIter {
        let mut rng = thread_rng();
        let mut indexes: Vec<_> = (0..s.read().unwrap().len() as i64).collect();
        indexes.shuffle(&mut rng);
        DatasetIter { dataset: s, indexes, batch_size }
    }
    pub fn iter(s: Arc<RwLock<Self>>, batch_size: usize) -> DatasetIter {
        let indexes: Vec<_> = (0..s.read().unwrap().len() as i64).collect();
        DatasetIter { dataset: s, indexes, batch_size }
    }

    pub fn len(&self) -> usize {
        self.samples.lock().unwrap().size()[0] as usize
    }
}

impl TryFrom<SizedObjectsBytes> for Dataset {
    type Error = TchError;

    fn try_from(value: SizedObjectsBytes) -> Result<Self, Self::Error> {
        let dataset = Dataset { samples: Mutex::new(Tensor::of_slice::<f32>(&[])), labels: Mutex::new(Tensor::of_slice::<f32>(&[])) };
        for object in value {
            let data = Tensor::load_multi_from_stream_with_device(Cursor::new(object), Device::Cpu)?;
            for (name, tensor) in data {
                let mut samples = dataset.samples.lock().unwrap();
                let mut labels = dataset.labels.lock().unwrap();
                match &*name {
                    "samples" => *samples = Tensor::f_cat(&[&*samples, &tensor], 0)?,
                    "labels" => *labels = Tensor::f_cat(&[&*labels, &tensor], 0)?,
                    s => return Err(TchError::FileFormat(String::from(format!("Invalid data, unknown field {}.", s)))),
                };
            }
        }
        Ok(dataset)
    }
}

impl TryFrom<&Dataset> for SizedObjectsBytes {
    type Error = TchError;

    fn try_from(value: &Dataset) -> Result<Self, Self::Error> {
        let mut dataset_bytes = SizedObjectsBytes::new();
        let mut buf = Vec::new();
        Tensor::save_multi_to_stream(&[("samples", &*value.samples.lock().unwrap()), ("labels", &*value.labels.lock().unwrap())], &mut buf)?;
        dataset_bytes.append_back(buf);

        Ok(dataset_bytes)
    }
}

/// A loss function with average statistics
pub struct Metric {
    loss_fn: Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor, TchError> + Send>,
    value: f32,
    nb_samples: usize,
}

impl Metric {
    /// Returns a `Metric` corresponding to given name, if not available raises an error.
    pub fn try_from_name(loss_name: &str) -> Result<Self, TchError> {
        Ok(Metric {
            loss_fn: match loss_name {
                "accuracy" => Box::new(|output, label| {
                    let prediction = output.f_argmax(-1, false)?.double_value(&[]);
                    Ok(if prediction == label.double_value(&[]) {
                        Tensor::of_slice(&[1]).f_view([])?
                    } else {
                        Tensor::of_slice(&[1]).f_view([])?
                    })
                }),
                "l2" => Box::new(|output, label| l2_loss(output, label)),
                s => return Err(TchError::FileFormat(String::from(format!("Invalid loss name, unknown loss {}.", s)))),
            },
            value: 0.0,
            nb_samples: 1,
        })
    }

    /// Computes the metric's value given `output` and `label` and updates the average.
    pub fn compute(&mut self, output: &Tensor, label: &Tensor) -> Result<Tensor, TchError> {
        let loss = (self.loss_fn)(output, label)?;
        self.value += 1./(self.nb_samples as f32) * (loss.double_value(&[]) as f32 - self.value);
        self.nb_samples += 1;
        Ok(loss)
    }

    /// Returns the average.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Resets the metric
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.nb_samples = 1;
    }
}

/// Stored object with encryption and owner key
#[derive(Debug)]
pub struct Artifact<T> {
    pub data: Arc<RwLock<T>>,
    pub description: String,
    pub secret: hmac::Key,
}

impl<T> Artifact<T> {
    /// Creates new artifact from data, description and owner key.
    pub fn new(data: T, description: String, secret: &[u8]) -> Self {
        Artifact {
            data: Arc::new(RwLock::new(data)),
            description,
            secret: hmac::Key::new(hmac::HMAC_SHA256, &secret),
        }
    }

    /// Verifies passed meaasge and tag against stored owner key.
    pub fn verify(&self, msg: &[u8], tag: &[u8]) -> bool {
        match hmac::verify(&self.secret, msg, tag) {
            Ok(()) => true,
            Err(_) => false,
        }
    }
}

impl<T> Artifact<T>
where
    for<'a> &'a T: TryInto<SizedObjectsBytes, Error = TchError>,
{
    /// Serializes the contained object and returns a new artifact that contains
    /// a SizedObjectBytes instead of the object.
    /// 
    /// Note that the object should be convertible into a SizedObjectBytes (with `TryInto`).
    pub fn serialize(&self) -> Result<Artifact<SizedObjectsBytes>, TchError> {
        Ok(Artifact {
            data: Arc::new(RwLock::new((&*self.data.read().unwrap()).try_into()?)),
            description: self.description.clone(),
            secret: self.secret.clone(),
        })
    }
}

impl Artifact<SizedObjectsBytes> {
    /// Deserializes the contained [`SizedObjectBytes`] object and returns a new artifact that contains
    /// a the deserialized object instead.
    /// 
    /// Note that the object should be convertible from a SizedObjectBytes (with `TryForm`).
    pub fn deserialize<T: TryFrom<SizedObjectsBytes, Error = TchError> + std::fmt::Debug>(
        self,
    ) -> Result<Artifact<T>, TchError> {
        Ok(Artifact {
            data: Arc::new(RwLock::new(T::try_from(Arc::try_unwrap(self.data).unwrap().into_inner().unwrap())?)),
            description: self.description,
            secret: self.secret,
        })
    }
}
