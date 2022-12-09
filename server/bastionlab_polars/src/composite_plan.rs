use base64;
use bytes::Bytes;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tch::CModule;
use tonic::Status;

use crate::{
    access_control::{Context, Policy},
    utils::*,
    visitable::{Visitable, VisitableMut},
    BastionLabPolars, DataFrameArtifact,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct CompositePlan(Vec<CompositePlanSegment>);

#[derive(Debug, Serialize, Deserialize)]
pub enum CompositePlanSegment {
    PolarsPlanSegment(LogicalPlan),
    UdfPlanSegment { columns: Vec<String>, udf: String },
    EntryPointPlanSegment(String),
    StackPlanSegment,
}

impl CompositePlan {
    pub fn run(self, state: &BastionLabPolars, token: Option<Bytes>) -> Result<DataFrameArtifact, Status> {
        let mut input_dfs = Vec::new();
        let plan_str = serde_json::to_string(&self.0).unwrap(); // FIX THIS

        let (policy, blacklist) = self.output_policy(state)?;
        let mut min_agg_size = None;

        for seg in self.0 {
            match seg {
                CompositePlanSegment::PolarsPlanSegment(mut plan) => {
                    initialize_inputs(&mut plan, &mut input_dfs)?;
                    aggregation_size(&plan, &mut min_agg_size)?;
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
                CompositePlanSegment::StackPlanSegment => {
                    let mut df1 = input_dfs.pop().ok_or(Status::invalid_argument(
                        "Could not apply stack: no input data frame",
                    ))?;

                    let df2 = input_dfs.pop().ok_or(Status::invalid_argument(
                        "Could not apply stack: no df2 input data frame",
                    ))?;

                    df1 = df1.vstack(&df2).map_err(|e| {
                        Status::invalid_argument(format!("Error while running vstack: {}", e))
                    })?;
                    input_dfs.push(df1);
                }
            }
        }

        if input_dfs.len() != 1 {
            return Err(Status::invalid_argument(
                "Wrong number of input data frames",
            ));
        }
        let sessions = state.sess_manager.sessions.read().unwrap();
        let token = match &token {
            Some(v) => &v[..],
            None => &[0u8; 32],
        };
        let session = sessions.get(token).ok_or(Status::aborted("Session not found!"))?;
        Ok(DataFrameArtifact {
            dataframe: input_dfs.pop().unwrap(),
            fetchable: policy.verify(&Context {
                min_agg_size,
                user_id: session.pubkey.clone(),
            })?,
            policy,
            blacklist,
            query_details: plan_str,
        })
    }

    fn output_policy(&self, state: &BastionLabPolars) -> Result<(Policy, Vec<String>), Status> {
        let mut policy = Policy::allow_by_default();
        let mut blacklist = Vec::new();

        for seg in &self.0 {
            if let CompositePlanSegment::EntryPointPlanSegment(identifier) = seg {
                state.with_df_artifact_ref(&identifier, |artifact| {
                    policy = policy.merge(&artifact.policy);
                    blacklist.extend_from_slice(&artifact.blacklist[..]);
                })?;
            }
        }

        Ok((policy, blacklist))
    }
}

fn aggregation_size(plan: &LogicalPlan, state: &mut Option<usize>) -> Result<(), Status> {
    plan.visit(state, |plan, state| {
        match plan {
            LogicalPlan::Aggregate { input, keys, .. } => {
                let keys = &(**keys)[..];
                let ldf = lazy_frame_from_logical_plan((&**input).clone());
                let agg_size: usize = ldf
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
                *state = match *state {
                    Some(prev_agg_size) => Some(prev_agg_size.min(agg_size)),
                    None => Some(agg_size),
                };
            }
            LogicalPlan::Join { .. } => *state = None,
            // These are not currently supported
            // LogicalPlan::ExtContext { .. } => *state = false,
            // LogicalPlan::Union { .. } => *state = false,
            _ => (),
        }
        Ok(())
    })
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
