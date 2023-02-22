use std::error::Error;

use bastionlab_common::array_store::ArrayStore;
use linfa::{prelude::Records, DatasetBase, Label};
use tonic::{Request, Status};

use crate::linfa_proto::{Trainer, TrainingRequest};

#[derive(Debug)]
pub struct IArrayStore(pub ArrayStore);

impl Records for IArrayStore {
    type Elem = IArrayStore;

    fn nsamples(&self) -> usize {
        self.0.height()
    }

    fn nfeatures(&self) -> usize {
        self.0.width()
    }
}

#[derive(Debug, Hash, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord)]
/// Replace [`u64`] in targets in [`DatasetBase`] because linfa fails to implement [`u64`] for Label
pub struct LabelU64(pub u64);

impl Label for LabelU64 {}

pub fn get_datasets(
    records: ArrayStore,
    target: ArrayStore,
) -> DatasetBase<IArrayStore, IArrayStore> {
    let dataset = DatasetBase::new(IArrayStore(records), IArrayStore(target));

    dataset
}

pub fn process_trainer_req(
    request: Request<TrainingRequest>,
) -> Result<(String, String, Option<Trainer>), Status> {
    Ok((
        request.get_ref().records.clone(),
        request.get_ref().target.clone(),
        Some(
            request
                .get_ref()
                .trainer
                .clone()
                .ok_or_else(|| Status::aborted("Trainer not supported!"))?,
        ),
    ))
}

/// This method is used in [`get_inner_array`] to emit an error during transformation
pub fn failed_array_type<T, A: std::fmt::Debug>(model: &str, array: T) -> Result<A, Status>
where
    T: std::fmt::Debug,
{
    return Err(Status::failed_precondition(format!(
        "{model} received wrong array type: ArrayStore(ArrayBase<{:?}>)",
        array
    )));
}

/// This method is used in [`get_inner_array`] as well to emit an error during dimensionality conversion
/// i.e., converting IxDynImpl into Ix
pub fn dimensionality_error<E: Error>(dim: &str, e: E) -> Status {
    return Status::aborted(format!(
        "Could not convert Dynamic Array into {:?}: {e}",
        dim
    ));
}
