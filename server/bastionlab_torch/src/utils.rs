use bastionlab_common::session_proto::TensorMetaData;
use serde::{Deserialize, Serialize};
use tch::{Kind, TchError, Tensor};
use tonic::Status;

use crate::torch_proto::RemoteDatasetReference;

/// Converts a [`tch::TchError`]-based result into a [`tonic::Status`]-based one.
pub fn tcherror_to_status<T>(input: Result<T, TchError>) -> Result<T, Status> {
    input.map_err(|err| Status::internal(format!("Torch error: {}", err)))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoteTensor {
    pub identifier: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoteDataset {
    pub inputs: Vec<RemoteTensor>,
    pub labels: RemoteTensor,
    pub privacy_limit: f64,
}

impl From<RemoteDatasetReference> for RemoteDataset {
    fn from(dataset: RemoteDatasetReference) -> Self {
        let inputs = dataset
            .inputs
            .iter()
            .map(|i| RemoteTensor {
                identifier: i.identifier.clone(),
            })
            .collect::<Vec<_>>();

        let labels = RemoteTensor {
            identifier: dataset.labels.unwrap().identifier,
        };

        Self {
            inputs,
            labels,
            privacy_limit: -1.0,
        }
    }
}

pub fn get_kind(kind: &str) -> Result<Kind, Status> {
    match kind {
        "Uint8" => Ok(Kind::Uint8),
        "Int8" => Ok(Kind::Int8),
        "Int16" => Ok(Kind::Int16),
        "Int" => Ok(Kind::Int),
        "Int64" => Ok(Kind::Int64),
        "Half" => Ok(Kind::Half),
        "Float" => Ok(Kind::Float),
        "Float32" => Ok(Kind::Float),
        "Float64" => Ok(Kind::Double),
        "Double" => Ok(Kind::Double),
        "ComplexHalf" => Ok(Kind::ComplexHalf),
        "ComplexFloat" => Ok(Kind::ComplexFloat),
        "ComplexDouble" => Ok(Kind::ComplexDouble),
        "Bool" => Ok(Kind::Bool),
        "QInt8" => Ok(Kind::QInt8),
        "QUInt8" => Ok(Kind::QUInt8),
        "QInt32" => Ok(Kind::QInt32),
        "BFloat16" => Ok(Kind::BFloat16),
        &_ => {
            return Err(Status::failed_precondition(format!(
                "Unsupported Kind: {}",
                kind
            )));
        }
    }
}

pub fn create_tensor_meta(tensor: &Tensor) -> TensorMetaData {
    TensorMetaData {
        input_dtype: vec![format!("{:?}", tensor.kind())],
        input_shape: tensor.size(),
    }
}
