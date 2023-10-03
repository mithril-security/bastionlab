use std::{error::Error, fmt::Display};

use bastionlab_common::array_store::{ArrayStore, ArrayStoreType};
use linfa::{
    prelude::{Records, SingleTargetRegression, ToConfusionMatrix},
    DatasetBase, Label,
};
use ndarray::Ix2;
use polars::{
    prelude::{DataFrame, NamedFrom},
    series::Series,
};
use tonic::{Request, Status};

use crate::linfa_proto::{
    classification_metric::ClassificationMetric, regression_metric::RegressionMetric,
    simple_validation_request::Scoring, Trainer, TrainingRequest,
};

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
    ($variant:tt, $array:ident, $dim:ty, $dim_str:tt, $model_name:tt, $inner:tt) => {{
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

impl Display for RegressionMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metric_name = match self {
            RegressionMetric::R2Score(_) => "r2_score",
            RegressionMetric::MaxError(_) => "max_error",
            RegressionMetric::MeanAbsoluteError(_) => "mean_absolute_error",
            RegressionMetric::MeanSquaredLogError(_) => "mean_squared_log_error",
            RegressionMetric::MeanSquaredError(_) => "mean_squared_error",
            RegressionMetric::ExplainedVariance(_) => "explained_variance",
            RegressionMetric::MedianAbsoluteError(_) => "mean_absolute_error",
        };
        write!(f, "{}", metric_name)
    }
}

impl Display for ClassificationMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metric_name = match self {
            ClassificationMetric::Accuracy(_) => "accuracy_score",
            ClassificationMetric::F1Score(_) => "f1_score",
            ClassificationMetric::Mcc(_) => "mcc_score",
        };
        write!(f, "{}", metric_name)
    }
}

impl RegressionMetric {
    pub fn compute(
        &self,
        prediction: IArrayStore,
        truth: IArrayStore,
    ) -> Result<DataFrame, Status> {
        let scoring = self.to_string().to_lowercase();

        let truth = get_inner_array! {AxdynF64, truth, Ix2, "Ix2", scoring, "Prediction"};
        let prediction = get_inner_array! {AxdynF64, prediction, Ix2, "Ix2",scoring, "Truth"};

        // Similar to what we did [`process_train_data`], we will first convert the array into Ix2 and then pick the first column
        // which is the same as reshaping but less expensive if there's only one column

        // We leave this as column without `to_owned` because `regression_metrics` accepts `ViewRepr` for truth.
        let truth = truth.column(0);

        let prediction = prediction.column(0).to_owned();

        let scores = match self {
            RegressionMetric::R2Score(_) => prediction.r2(&truth),
            RegressionMetric::MaxError(_) => prediction.max_error(&truth),
            RegressionMetric::MeanAbsoluteError(_) => prediction.mean_absolute_error(&truth),
            RegressionMetric::ExplainedVariance(_) => prediction.explained_variance(&truth),
            RegressionMetric::MeanSquaredLogError(_) => prediction.mean_squared_log_error(&truth),
            RegressionMetric::MedianAbsoluteError(_) => prediction.median_absolute_error(&truth),
            RegressionMetric::MeanSquaredError(_) => prediction.mean_squared_error(&truth),
        };

        let scores = scores.map_err(|e| {
            Status::internal(format!(
                "Could not compute metric {self:?} for {prediction:?} and {truth:?}: {e}"
            ))
        })?;
        let df = Series::new(&scoring, [scores]).into_frame();
        Ok(df)
    }
}

impl ClassificationMetric {
    pub fn compute(
        &self,
        prediction: IArrayStore,
        truth: IArrayStore,
    ) -> Result<DataFrame, Status> {
        let scoring = format!("{self:?}").to_lowercase();
        let name = &format!("ClassificationMetrics: {scoring}");

        // These casts are to prevent the user from having to cast predictions and truth arrays
        let truth = truth.cast(ArrayStoreType::UInt64)?;
        let prediction = prediction.cast(ArrayStoreType::UInt64)?;

        let truth = get_inner_array!(AxdynU64, truth, Ix2, "Ix2", name, "Prediction");
        let prediction = get_inner_array!(AxdynU64, prediction, Ix2, "Ix2", name, "Truth");

        // Similar to what we did [`process_train_data`], we will first convert the array into Ix2 and then pick the first column
        // which is the same as reshaping but less expensive if there's only one column

        // We leave this as column without `to_owned` because `regression_metrics` accepts `ViewRepr` for truth.
        let truth = truth.column(0);

        let prediction = prediction.column(0).to_owned();

        let prediction = prediction.map(|v| LabelU64(*v));
        let truth = truth.map(|v| LabelU64(*v));

        let cm = prediction
            .confusion_matrix(truth)
            .map_err(|e| Status::internal(format!("Could not compute Confusion Matrix: {e}")))?;

        let scores = match self {
            ClassificationMetric::Accuracy(_) => cm.accuracy() as f64,
            ClassificationMetric::F1Score(_) => cm.f1_score() as f64,
            ClassificationMetric::Mcc(_) => cm.mcc() as f64,
        };
        let df = Series::new(&scoring, [scores]).into_frame();
        Ok(df)
    }
}

pub fn get_score(
    scoring: &Scoring,
    prediction: IArrayStore,
    truth: IArrayStore,
) -> Result<DataFrame, Status> {
    match scoring {
        Scoring::RegressionMetric(metric) => metric
            .regression_metric
            .clone()
            .unwrap()
            .compute(prediction, truth),
        Scoring::ClassificationMetric(metric) => metric
            .classification_metric
            .clone()
            .unwrap()
            .compute(prediction, truth),
    }
}
