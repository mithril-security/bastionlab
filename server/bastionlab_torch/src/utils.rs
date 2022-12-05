use tch::TchError;
use tonic::Status;

/// Converts a [`tch::TchError`]-based result into a [`tonic::Status`]-based one.
pub fn tcherror_to_status<T>(input: Result<T, TchError>) -> Result<T, Status> {
    input.map_err(|err| Status::internal(format!("Torch error: {}", err)))
}
