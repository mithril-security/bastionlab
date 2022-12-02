use std::error::Error;

use tonic::Status;

pub mod algorithms;
pub mod operations;
pub mod trainer;

pub fn to_status_error<T, E: Error>(input: Result<T, E>) -> Result<T, Status> {
    input.map_err(|err| Status::aborted(err.to_string()))
}
