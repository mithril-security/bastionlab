use std::error::Error;

use bastionlab_common::array_store::{ArrayStore, ArrayStoreType};
use linfa::{
    prelude::{Records, SingleTargetRegression, ToConfusionMatrix},
    DatasetBase, Label,
};
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr, ViewRepr};
use polars::{
    prelude::{DataFrame, NamedFrom},
    series::Series,
};
use tonic::{Request, Status};

use crate::linfa_proto::{Trainer, TrainingRequest};

/*
----------------------
         MACROS
----------------------
*/

/// This macro converts convert the Dynamic Array Implementation into
/// a fixed dimension say `Ix2`.
///
/// It does this by first matching on the right enum variant (considering the type
///  of the array).
///
/// It calls `into_dimensionality` to pass the dimension as a type to the macro.
#[macro_export]
macro_rules! get_inner_array {
    ($variant:tt, $array:ident, $dim:ty, $dim_str:tt, $model_name:expr, $inner:tt) => {{
        use crate::utils::{dimensionality_error, failed_array_type, IArrayStore};
        match $array {
            IArrayStore(ArrayStore::$variant(a)) => {
                let a = a
                    .into_dimensionality::<$dim>()
                    .map_err(|e| dimensionality_error(&format!("{:?}", $dim_str), e))?;
                a
            }
            _ => return failed_array_type(&format!("{} -> {}", $model_name, $inner), $array),
        }
    }};
}

/// This macro converts `DatasetBase<IArrayBase>` into `DatasetBase<ArrayBase<T, Ix...>>`
///
#[macro_export]
macro_rules! prepare_train_data {
    ($model:tt, $train:ident, ($t_variant:tt, $t_dim:ty)) => {{
        let records = $train.records;
        let targets = $train.targets;
        let records = get_inner_array! {AxdynF64, records, Ix2, "Ix2", $model, "Records"};

        /*
            Intuitively, we ought to convert targets directly into a Ix1 but Polars' `DataFrame -> ndarray`
            conversion only uses `Array2`.

            We will have to first convert it from `DynImpl` into `Ix2` then later reshape into `Ix1`.

            Also, for the edge case of using `KMeans`, we will only choose the first column if there are multiple
            columns in the target array.
         */
        let targets = get_inner_array! {$t_variant, targets, Ix2, "Ix2", $model, "Targets"};

        // Select the first column
        let targets = targets.column(0).to_owned();

        // Here, we construct the specific DatasetBase with the right types
        DatasetBase::new(records, targets)
    }};
}

#[derive(Debug)]
pub struct IArrayStore(pub ArrayStore);

impl IArrayStore {
    /* Here, we will add regression and classification metrics on the IArrayStore, which essentially wraps
    around an ArrayStore.


    NB: We choose to implement them here because the regression methods we implemented in the `linfa` crate and are not
    directly on ArrayBase and so, we couldn't move it out of the `bastionlab_linfa` crate.

    In this case, we will be able to perform validation on the client.

    In that case, the linfa API for metrics validation will call these metrics methods on the ArrayStore directly
    instead of matching each ArrayBase type.
    */

    /* Regression metrics */
    pub fn regression_metrics(self, other: Self, scoring: &str) -> Result<DataFrame, Status> {
        let name = &format!("RegressionMetrics: {scoring}");

        let truth = get_inner_array! {AxdynF64, self, Ix2, "Ix2", name, "Prediction"};
        let pred = get_inner_array! {AxdynF64, other, Ix2, "Ix2",name, "Truth"};

        // Similar to what we did [`process_train_data`], we will first convert the array into Ix2 and then pick the first column
        // which is the same as reshaping but less expensive if there's only one column

        // We leave this as column without `to_owned` because `regression_metrics` accepts `ViewRepr` for truth.
        let truth = truth.column(0);

        let pred = pred.column(0).to_owned();

        let scores = regression_metrics(&pred, &truth, scoring).map_err(|e| {
            Status::aborted(format!(
                "Could not compute {scoring} on both prediction and truth: {e}"
            ))
        })?;

        let df = Series::new(scoring, [scores]).into_frame();
        Ok(df)
    }

    /* Classification metrics */
    pub fn classification_metrics(self, other: Self, scoring: &str) -> Result<DataFrame, Status> {
        let name = &format!("ClassificationMetrics: {scoring}");

        // These casts are to prevent the user from having to cast predictions and truth arrays
        let truth = self.cast(ArrayStoreType::UInt64)?;
        let pred = other.cast(ArrayStoreType::UInt64)?;

        let truth = get_inner_array!(AxdynU64, truth, Ix2, "Ix2", name, "Prediction");
        let pred = get_inner_array!(AxdynU64, pred, Ix2, "Ix2", name, "Truth");

        // Similar to what we did [`process_train_data`], we will first convert the array into Ix2 and then pick the first column
        // which is the same as reshaping but less expensive if there's only one column

        // We leave this as column without `to_owned` because `regression_metrics` accepts `ViewRepr` for truth.
        let truth = truth.column(0);

        let pred = pred.column(0).to_owned();

        let pred = pred.map(|v| LabelU64(*v));
        let truth = truth.map(|v| LabelU64(*v));

        let scores = classification_metrics(&pred, &truth.view(), scoring).map_err(|e| {
            Status::aborted(format!(
                "Could not compute {scoring} on both prediction and truth: {e}"
            ))
        })?;
        let df = Series::new(scoring, [scores]).into_frame();
        Ok(df)
    }
    pub fn cast(&self, dtype: ArrayStoreType) -> Result<Self, Status> {
        Ok(Self(self.0.cast(dtype)?))
    }
}

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
        "{model} received wrong array type: {:?}",
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

pub fn regression_metrics(
    prediction: &ArrayBase<OwnedRepr<f64>, Ix1>,
    truth: &ArrayBase<ViewRepr<&f64>, Ix1>,
    metric: &str,
) -> Result<f64, linfa::Error> {
    match metric {
        "r2_score" => prediction.r2(truth),
        "max_error" => prediction.max_error(truth),
        "mean_absolute_error" => prediction.mean_absolute_error(truth),
        "explained_variance" => prediction.explained_variance(truth),
        "mean_squared_log_error" => prediction.mean_squared_log_error(truth),
        "mean_squared_error" => prediction.mean_squared_error(truth),
        "median_absolute_error" => prediction.median_absolute_error(truth),
        _ => {
            return Err(linfa::Error::Priors(format!(
                "Unsupported metric: {}",
                metric
            )))
        }
    }
}

pub fn classification_metrics(
    prediction: &ArrayBase<OwnedRepr<LabelU64>, Ix1>,
    truth: &ArrayBase<ViewRepr<&LabelU64>, Ix1>,
    metric: &str,
) -> Result<f32, linfa::Error> {
    let cm = prediction.confusion_matrix(truth)?;
    match metric {
        "accuracy" => Ok(cm.accuracy()),
        "f1_score" => Ok(cm.f1_score()),
        "mcc" => Ok(cm.mcc()),
        _ => {
            return Err(linfa::Error::Priors(format!(
                "Could not find metric: {}",
                metric
            )))
        }
    }
}
