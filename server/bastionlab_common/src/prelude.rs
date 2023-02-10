pub use anyhow::{anyhow, bail, ensure, Context, Result};
pub use log::{debug, error, info, trace, warn};
pub use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, RwLock},
};

pub trait ResultExt<T>: Sized {
    fn as_status<F>(self, f: F) -> Result<T, tonic::Status>
    where
        F: FnOnce(String) -> tonic::Status;

    fn or_invalid_argument(self) -> Result<T, tonic::Status> {
        self.as_status(tonic::Status::invalid_argument)
    }
    fn or_internal_error(self) -> Result<T, tonic::Status> {
        self.as_status(tonic::Status::internal)
    }
}

impl<T> ResultExt<T> for Result<T, tch::TchError> {
    fn as_status<F>(self, f: F) -> Result<T, tonic::Status>
    where
        F: FnOnce(String) -> tonic::Status,
    {
        self.map_err(|err| format!("Torch error: {}", err))
            .map_err(f)
    }
}

impl<T> ResultExt<T> for Result<T, polars::prelude::PolarsError> {
    fn as_status<F>(self, f: F) -> Result<T, tonic::Status>
    where
        F: FnOnce(String) -> tonic::Status,
    {
        self.map_err(|err| format!("Polars error: {}", err))
            .map_err(f)
    }
}

impl<T> ResultExt<T> for Result<T, String> {
    fn as_status<F>(self, f: F) -> Result<T, tonic::Status>
    where
        F: FnOnce(String) -> tonic::Status,
    {
        self.map_err(f)
    }
}

impl<T> ResultExt<T> for Result<T, &'static str> {
    fn as_status<F>(self, f: F) -> Result<T, tonic::Status>
    where
        F: FnOnce(String) -> tonic::Status,
    {
        self.map_err(String::from).map_err(f)
    }
}
