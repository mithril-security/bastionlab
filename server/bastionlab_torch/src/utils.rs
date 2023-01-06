use bastionlab_learning::data::Dataset;
use serde::{Deserialize, Serialize};
use tch::TchError;
use tonic::Status;

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
    pub label: RemoteTensor,
    pub nb_samples: usize,
    pub privacy_limit: f64,
}
