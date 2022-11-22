use base64;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tch::CModule;
use tonic::Status;

use crate::{
    visitable::{Visitable, VisitableMut},
    BastionLabState, DataFrameArtifact, access_control::Policy,
    utils::*
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
        // let mut fetchable = Access::Denied(String::from("Denied by default"));
        let plan_str = serde_json::to_string(&self.0).unwrap(); // FIX THIS

        let policy = self.output_policy(state)?;
        let fetchable = policy.verify_fetch(&self, "")?;

        for seg in self.0 {
            match seg {
                CompositePlanSegment::PolarsPlanSegment(mut plan) => {
                    initialize_inputs(&mut plan, &mut input_dfs)?;
                    // aggregation_check(&plan, &mut fetchable, 10)?;
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
                    input_dfs.push(state.get_df_unchecked(&identifier)?);
                }
            }
        }

        if input_dfs.len() != 1 {
            return Err(Status::invalid_argument(
                "Wrong number of input data frames",
            ));
        }

        Ok(DataFrameArtifact {
            dataframe: input_dfs.pop().unwrap(),
            policy,
            fetchable,
            query_details: plan_str,
        })
    }

    fn output_policy(&self, state: &BastionLabState) -> Result<Policy, Status> {
        let mut policy = Policy::allow_by_default();

        for seg in &self.0 {
            if let CompositePlanSegment::EntryPointPlanSegment(identifier) = seg {
                policy = policy.merge(&state.get_policy_unchecked(&identifier)?);
            }
        }

        Ok(policy)
    }

    pub fn aggregation_match(&self, min_allowed_agg_size: usize) -> Result<bool, Status> {
        let mut state = false;

        for seg in &self.0 {
            if let CompositePlanSegment::PolarsPlanSegment(plan) = seg {
                plan.visit(&mut state, |plan, state| {
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
                })?;
            }
        }
        
        Ok(state)
    }
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

fn run_logical_plan(plan: LogicalPlan) -> Result<DataFrame, Status> {
    let ldf = lazy_frame_from_logical_plan(plan);
    ldf.collect()
        .map_err(|e| Status::internal(format!("Could not run logical plan: {}", e)))
}
