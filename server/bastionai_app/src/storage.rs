use ring::hmac;
use tch::{TchError, Device, IValue, Tensor, TrainableCModule, CModule};
use tch::nn::VarStore;
use tonic::{Streaming, Response};
use std::sync::Mutex;
use std::convert::{TryFrom, TryInto};
use std::collections::VecDeque;
use crate::utils::*;

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
        let len = read_le_usize(&mut &self.0.drain(..4).collect::<Vec<u8>>()[..]);
        self.0.drain(..len).collect()
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

pub struct Module {
    c_module: TrainableCModule,
    var_store: VarStore,
}

impl TryFrom<SizedObjectsBytes> for Module {
    type Error = TchError;

    fn try_from(mut value: SizedObjectsBytes) -> Result<Self, Self::Error> {
        let mut object = value.next().ok_or(TchError::FileFormat(String::from("Invalid data, expected at least one object in stream.")))?;
        let vs = VarStore::new(Device::Cpu);
        Ok(Module { c_module: TrainableCModule::load_data(&mut &object[..], vs.root())?, var_store: vs })
    }
}

impl TryFrom<&Module> for SizedObjectsBytes {
    type Error = TchError;

    fn try_from(value: &Module) -> Result<Self, Self::Error> {
        let parameters: Vec<(IValue, IValue)> = match value.c_module.method_is::<IValue>("trainable_parameters", &[])? {
            IValue::GenericDict(v) => Ok(v),
            _ => Err(TchError::FileFormat(String::from("Invalid data, expected module to have a `trainable_parameters` function returning the dict of named trainable parameters.")))
        }?;
        
        let mut parameters_bytes = SizedObjectsBytes::new();
        for (name, parameter) in parameters {
            let mut name_bytes = match name {
                IValue::String(s) => Ok(s.into_bytes()),
                _ => Err(TchError::FileFormat(String::from("Invalid data, expected value to be of type string in the dict returned by module's `trainable_parameters` function.")))
            }?;
            let mut parameter_bytes = match parameter {
                IValue::Tensor(t) => Ok(serialize_tensor(&t)),
                _ => Err(TchError::FileFormat(String::from("Invalid data, expected value to be of type tensor in the dict returned by module's `trainable_parameters` function.")))
            }?;
            parameters_bytes.append_back(name_bytes);
            parameters_bytes.append_back(parameter_bytes);
        }
        
        Ok(parameters_bytes)
    }
}

pub struct Dataset(Vec<Mutex<Tensor>>);

impl TryFrom<SizedObjectsBytes> for Dataset {
    type Error = TchError;

    fn try_from(mut value: SizedObjectsBytes) -> Result<Self, Self::Error> {
        let mut dataset = Dataset(Vec::new());
        for mut object in value {
            let module = CModule::load_data(&mut &object[..])?;
            match module.method_is::<IValue>("data", &[])? {
                IValue::TensorList(v) => dataset.0.append(&mut v.into_iter().map(|x| Mutex::new(x)).collect()),
                _ => return Err(TchError::FileFormat(String::from("Invalid data, expected a batch module with a `data` function returning the actual data."))),
            }
        }

        Ok(dataset)
    }
}

impl TryFrom<&Dataset> for SizedObjectsBytes {
    type Error = TchError;

    fn try_from(value: &Dataset) -> Result<Self, Self::Error> {
        let mut tensors_bytes = SizedObjectsBytes::new();
        for tensor in value.0.iter() {
            let bytes = serialize_tensor(&tensor.lock().unwrap());
            tensors_bytes.append_back(bytes);
        }
        
        Ok(tensors_bytes)
    }
}

// pub enum ArtifactData {
//     Module(Module),
//     Dataset(Dataset),
// }

// impl From<&ArtifactData> for SizedObjectsBytes {
//     fn from(value: &ArtifactData) -> Self {
//         match value {
//             ArtifactData::Module(m) => m.into(),
//             ArtifactData::Dataset(d) => d.into(),
//         }
//     }
// }

pub struct Artifact<T> {
    pub data: T,
    pub description: String,
    pub secret: hmac::Key,
}

impl<T> Artifact<T> {
    pub fn new(data: T, description: String, secret: &[u8]) -> Self {
        Artifact {
            data,
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
    for<'a> &'a T: TryInto<SizedObjectsBytes, Error=TchError>
{
    pub fn serialize(&self) -> Result<Artifact<SizedObjectsBytes>, TchError> {
        Ok(Artifact {
            data: (&self.data).try_into()?,
            description: self.description.clone(),
            secret: self.secret.clone(),
        })
    }
}

impl Artifact<SizedObjectsBytes> {
    pub fn deserialize<T: TryFrom<SizedObjectsBytes, Error=TchError>>(mut self) -> Result<Artifact<T>, TchError> {
        Ok(Artifact {
            data: T::try_from(self.data)?,
            description: self.description,
            secret: self.secret,
        })
    }

    // pub fn deserialize_dataset(self) -> Result<Artifact<ArtifactData>, Status> {
    //     let dataset = tcherror_to_status(Dataset::try_from(self.data.next().into()))?;
    //     Artifact {
    //         data: ArtifactData::Dataset(dataset),
    //         description: self.description,
    //         secret: self.secret,
    //     }
    // }
}
