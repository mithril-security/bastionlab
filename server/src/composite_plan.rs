use base64;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tch::{kind::Element, CModule, Tensor};
use tonic::Status;

use crate::{
    visitable::{Visitable, VisitableMut},
    BastionLabState, DataFrameArtifact,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct CompositePlan(Vec<CompositePlanSegment>);

#[derive(Debug, Serialize, Deserialize)]
pub enum CompositePlanSegment {
    PolarsPlanSegment(LogicalPlan),
    UdfPlanSegment { columns: Vec<String>, udf: String },
    EntryPointPlanSegment(String),
}

impl CompositePlan {
    pub fn run(self, state: &BastionLabState) -> Result<DataFrameArtifact, Status> {
        let mut input_dfs = Vec::new();
        let mut has_aggregation = false;
        let plan_str = serde_json::to_string(&self.0).unwrap(); // FIX THIS
        for seg in self.0 {
            match seg {
                CompositePlanSegment::PolarsPlanSegment(mut plan) => {
                    initialize_inputs(&mut plan, &mut input_dfs)?;
                    aggregation_check(&plan, &mut has_aggregation, 10)?;
                    let df = run_logical_plan(plan)?;
                    input_dfs.push(df);
                }
                CompositePlanSegment::UdfPlanSegment { columns, udf } => {
                    let module =
                        CModule::load_data(&mut Cursor::new(base64::decode(udf).map_err(|e| {
                            Status::invalid_argument(format!(
                                "Could not decode bas64-encoded udf: {}",
                                e
                            ))
                        })?))
                        .map_err(|e| {
                            Status::invalid_argument(format!(
                                "Could not deserialize udf from bytes: {}",
                                e
                            ))
                        })?;

                    let mut df = input_dfs.pop().ok_or(Status::invalid_argument(
                        "Could not apply udf: no input data frame",
                    ))?;
                    for name in columns {
                        let idx = df
                            .get_column_names()
                            .iter()
                            .position(|x| x == &&name)
                            .ok_or(Status::invalid_argument(format!(
                                "Could not apply udf: no column `{}` in data frame",
                                name
                            )))?;
                        let series = df.get_columns_mut().get_mut(idx).unwrap();
                        let tensor = series_to_tensor(series)?;
                        let tensor = module.forward_ts(&[tensor]).map_err(|e| {
                            Status::invalid_argument(format!("Error while running udf: {}", e))
                        })?;
                        *series = tensor_to_series(series.name(), series.dtype(), tensor)?;
                    }
                    input_dfs.push(df);
                }
                CompositePlanSegment::EntryPointPlanSegment(identifier) => {
                    input_dfs.push(state.get_df_unchecked(&identifier)?)
                }
            }
        }

        if input_dfs.len() != 1 {
            return Err(Status::invalid_argument(
                "Wrong number of input data frames",
            ));
        }

        Ok(DataFrameArtifact { dataframe: input_dfs.pop().unwrap(), fetchable: has_aggregation, query_details: plan_str })
    }
}

fn series_to_tensor(series: &Series) -> Result<Tensor, Status> {
    Ok(match series.dtype() {
        DataType::Float32 => array_to_tensor(series.f32().unwrap())?,
        DataType::Float64 => array_to_tensor(series.f64().unwrap())?,
        DataType::Int64 => array_to_tensor(series.i64().unwrap())?,
        DataType::Int32 => array_to_tensor(series.i32().unwrap())?,
        DataType::Int16 => array_to_tensor(series.i16().unwrap())?,
        DataType::Int8 => array_to_tensor(series.i8().unwrap())?,
        d => {
            return Err(Status::invalid_argument(format!(
                "Unsuported data type in udf: {}",
                d
            )))
        }
    })
}

fn tensor_to_series(name: &str, dtype: &DataType, tensor: Tensor) -> Result<Series, Status> {
    Ok(match dtype {
        DataType::Float32 => Series::from(tensor_to_array::<Float32Type>(&name, tensor)),
        DataType::Float64 => Series::from(tensor_to_array::<Float64Type>(&name, tensor)),
        DataType::Int64 => Series::from(tensor_to_array::<Int64Type>(&name, tensor)),
        DataType::Int32 => Series::from(tensor_to_array::<Int32Type>(&name, tensor)),
        DataType::Int16 => Series::from(tensor_to_array::<Int16Type>(&name, tensor)),
        DataType::Int8 => Series::from(tensor_to_array::<Int8Type>(&name, tensor)),
        d => {
            return Err(Status::invalid_argument(format!(
                "Unsuported data type in udf: {}",
                d
            )))
        }
    })
}

fn array_to_tensor<T>(series: &ChunkedArray<T>) -> Result<Tensor, Status>
where
    T: PolarsNumericType,
    T::Native: Element,
{
    Ok(match series.rechunk().cont_slice() {
        Ok(slice) => Tensor::from(slice),
        Err(_) => {
            if !series.has_validity() {
                return Err(Status::invalid_argument(
                    "Cannot apply udf on a column that contains empty values",
                ));
            }
            let v: Vec<T::Native> = series.into_no_null_iter().collect();
            Tensor::from(&v[..])
        }
    })
}

fn tensor_to_array<T>(name: &str, tensor: Tensor) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Element,
{
    let v = Vec::from(tensor);
    ChunkedArray::new_vec(name, v)
}

fn lazy_frame_from_logical_plan(plan: LogicalPlan) -> LazyFrame {
    let mut ldf = LazyFrame::default();
    ldf.logical_plan = plan;
    ldf
}

fn run_logical_plan(plan: LogicalPlan) -> Result<DataFrame, Status> {
    let ldf = lazy_frame_from_logical_plan(plan);
    ldf.collect()
        .map_err(|e| Status::internal(format!("Could not run logical plan: {}", e)))
}

fn initialize_inputs(plan: &mut LogicalPlan, input_dfs: &mut Vec<DataFrame>) -> Result<(), Status> {
    plan.visit_mut(input_dfs, |plan, input_dfs| {
        if let LogicalPlan::DataFrameScan { .. } = plan {
            *plan = input_dfs
                .pop()
                .ok_or(Status::invalid_argument(
                    "Could not run logical plan: not enough input data frames",
                ))?
                .lazy()
                .logical_plan;
        }
        Ok(())
    })
}

fn aggregation_check(
    plan: &LogicalPlan,
    state: &mut bool,
    min_allowed_agg_size: usize,
) -> Result<(), Status> {
    plan.visit(state, |plan, state| {
        match plan {
            LogicalPlan::Aggregate { input, keys, .. } => {
                let keys = &(**keys)[..];
                let ldf = lazy_frame_from_logical_plan((&**input).clone());
                let min_agg_size: usize = ldf
                    .cache()
                    .with_row_count("__count", None)
                    .groupby(keys)
                    .agg([col("__count").count()])
                    .select([col("__count").min()])
                    .collect()
                    .map_err(|e| {
                        Status::internal(format!(
                            "Could not check aggregation minimal count: {}",
                            e
                        ))
                    })?
                    .get(0)
                    .unwrap()[0]
                    .try_extract()
                    .unwrap();
                *state = *state || min_agg_size >= min_allowed_agg_size;
            }
            LogicalPlan::Join { .. } => *state = false,
            // These are not currently supported
            // LogicalPlan::ExtContext { .. } => *state = false,
            // LogicalPlan::Union { .. } => *state = false,
            _ => (),
        }
        Ok(())
    })
}
