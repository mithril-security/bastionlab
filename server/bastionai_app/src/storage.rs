use crate::remote_torch::{TrainConfig, TestConfig, Accuracy};
use crate::utils::*;
use ring::hmac;
use std::collections::{VecDeque, HashMap};
use std::convert::{TryFrom, TryInto};
use std::io::Cursor;
use std::sync::Mutex;
use tch::nn::VarStore;
use tch::{CModule, Device, IValue, TchError, Tensor, TrainableCModule, IndexOp};
use private_learning::{Parameters, PrivateParameters, LossType, SGD, Optimizer, l2_loss};
use std::sync::RwLock;

#[derive(Debug)]
pub struct SizedObjectsBytes(Vec<u8>);

impl SizedObjectsBytes {
    pub fn new() -> Self {
        SizedObjectsBytes(Vec::new())
    }

    pub fn append_back(&mut self, mut bytes: Vec<u8>) {
        self.0.extend_from_slice(&bytes.len().to_le_bytes());
        self.0.append(&mut bytes);
    }

    pub fn remove_front(&mut self) -> Vec<u8> {
        let len = read_le_usize(&mut &self.0.drain(..8).collect::<Vec<u8>>()[..]);
        self.0.drain(..len).collect()
    }

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

#[derive(Debug)]
pub struct Module {
    c_module: TrainableCModule,
    var_store: VarStore,
}

impl Module {
    pub fn parameters(&self) -> Parameters {
        Parameters::new(&self.var_store)
    }
    pub fn private_parameters(&self, max_grad_norm: f64, noise_multiplier: f64, loss_type: LossType) -> PrivateParameters {
        PrivateParameters::new(&self.var_store, max_grad_norm, noise_multiplier, loss_type)
    }
    pub fn set_device(&mut self, device: Device) {
        self.var_store.set_device(device);
    }
    pub fn train(&mut self, dataset: &Dataset, config: TrainConfig, device: Device) -> Result<(), TchError> {
        self.set_device(device);
        let mut optimizer = if config.private_learning {
            let parameters = self.private_parameters(1.0, 0.01, private_learning::LossType::Sum);
            Box::new(SGD::new(parameters, config.learning_rate as f64)) as Box<dyn Optimizer>
        } else {
            let parameters = self.parameters();
            Box::new(SGD::new(parameters, config.learning_rate as f64)) as Box<dyn Optimizer>
        };
        for _ in 0..config.epochs {
            for (input, label) in dataset.iter() {
                let input = input.f_view([1, 1])?.f_to(device)?;
                let output = self.c_module.forward_ts(&[input])?;
                let loss = l2_loss(&output, &label)?;
                optimizer.zero_grad()?;
                loss.backward();
                optimizer.step()?;
            }
        }
        Ok(())
    }
    pub fn test(&self, dataset: &Dataset, config: TestConfig) -> Result<f32, TchError> {
        let mut count = 0;
        let mut correct = 0;
        for (input, label) in dataset.iter() {
            let input = input.f_view([1, 1])?;
            let output = self.c_module.forward_ts(&[input])?;
            let prediction = output.f_argmax(-1, false)?.double_value(&[]);
            if prediction == label.double_value(&[]) {
                correct += 1;
            }
            count += 1;
        }
        Ok(correct as f32 / count as f32)
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

pub enum Metric {
    Accuracy(usize, f32),
    Loss(fn(&Tensor, &Tensor) -> Result<Tensor, TchError>, usize, f32),
}

impl Metric {
    pub fn accumulate(&self, output: &Tensor, label: &Tensor) -> Result<(), TchError> {
        match self {
            Self::Accuracy(mut count, mut acc) => {
                let prediction = output.f_argmax(-1, false)?.double_value(&[]);
                if prediction == label.double_value(&[]) {
                    acc += 1./(count as f32) * (1. - acc);
                }
                count += 1;
            }
            Self::Loss(f, mut count, mut loss) => {
                loss += 1./(count as f32) * (f(output, label)?.double_value(&[]) as f32 - loss);
                count += 1;
            }
        }
        Ok(())
    }
    pub fn into_float(self) -> f32 {
        match self {
            Self::Accuracy(_, acc) => acc,
            Self::Loss(_, _, loss) => loss,
        }
    }
}

#[derive(Debug)]
pub struct Dataset {
    samples: Mutex<Tensor>,
    labels: Mutex<Tensor>,
}

pub struct DatasetIter<'a> {
    dataset: &'a Dataset,
    index: i64,
    len: i64,
}

impl<'a> Iterator for DatasetIter<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let samples = self.dataset.samples.lock().unwrap();
            let labels = self.dataset.labels.lock().unwrap();
            let res = Some((samples.i(self.index), labels.i(self.index)));
            self.index += 1;
            res
        } else {
            None
        }
    }
}

impl Dataset {
    pub fn iter(&self) -> DatasetIter<'_> {
        DatasetIter { dataset: &self, index: 0, len: self.samples.lock().unwrap().size()[0] }
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

// pub struct BatchedDataset {
//     index_queue: Mutex<VecDeque<usize>>,
//     data_store: Mutex<HashMap<usize, >>
// }

#[derive(Debug)]
pub struct Artifact<T> {
    pub data: RwLock<T>,
    pub description: String,
    pub secret: hmac::Key,
}

impl<T> Artifact<T> {
    pub fn new(data: T, description: String, secret: &[u8]) -> Self {
        Artifact {
            data: RwLock::new(data),
            description,
            secret: hmac::Key::new(hmac::HMAC_SHA256, &secret),
        }
    }

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
    pub fn serialize(&self) -> Result<Artifact<SizedObjectsBytes>, TchError> {
        Ok(Artifact {
            data: RwLock::new((&*self.data.read().unwrap()).try_into()?),
            description: self.description.clone(),
            secret: self.secret.clone(),
        })
    }
}

impl Artifact<SizedObjectsBytes> {
    pub fn deserialize<T: TryFrom<SizedObjectsBytes, Error = TchError> + std::fmt::Debug>(
        self,
    ) -> Result<Artifact<T>, TchError> {
        Ok(Artifact {
            data: RwLock::new(T::try_from(self.data.into_inner().unwrap())?),
            description: self.description,
            secret: self.secret,
        })
    }
}
