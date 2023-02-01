use base64;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tch::CModule;
use tonic::Status;

use crate::{
    access_control::{Context, Policy, VerificationResult},
    differential_privacy::{DPAnalyzerBasePass, PrivacyStats},
    init::Initializer,
    state_tree::StateTreeBuilder,
    utils::*,
    visit::{Visitor, VisitorMut},
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

#[derive(Debug)]
pub struct ExtendedDataFrame<S> {
    pub df: DataFrame,
    pub extension: S,
}

impl<S> ExtendedDataFrame<S> {
    fn new(df: DataFrame, extension: S) -> Self {
        ExtendedDataFrame { df, extension }
    }
}

pub struct Extension {
    privacy_stats: PrivacyStats,
}

impl Extension {
    fn new(privacy_stats: PrivacyStats) -> Self {
        Extension { privacy_stats }
    }
}

impl CompositePlan {
    pub fn run(self, state: &BastionLabPolars, user_id: &str) -> Result<DataFrameArtifact, Status> {
        let mut stack = Vec::new();
        let plan_str = serde_json::to_string(&self.0).map_err(|e| {
            Status::invalid_argument(format!("Could not parse composite plan: {e}"))
        })?;

        for seg in self.0 {
            match seg {
                CompositePlanSegment::PolarsPlanSegment(mut plan) => {
                    // Initialize logical plan
                    let mut init = Initializer::new(&mut stack);
                    polars_to_status_error(init.visit_logical_plan_mut(&mut plan))?;

                    let initial_states = init.initial_states();

                    // Differential Privacy Analysis
                    let dp_initial_states: Vec<_> = initial_states
                        .into_iter()
                        .map(|s: Extension| s.privacy_stats)
                        .collect();

                    let mut registry_builder =
                        StateTreeBuilder::from_initial_states(dp_initial_states);

                    polars_to_status_error(registry_builder.visit_logical_plan(&plan))?;

                    let mut cache_builder = StateTreeBuilder::new();
                    polars_to_status_error(cache_builder.visit_logical_plan(&plan))?;

                    let mut dp_analyzer = DPAnalyzerBasePass::new(
                        polars_to_status_error(registry_builder.state_tree())?,
                        polars_to_status_error(cache_builder.state_tree())?,
                    );

                    polars_to_status_error(dp_analyzer.visit_logical_plan(&plan))?;
                    let (stats, mut cache) = polars_to_status_error(dp_analyzer.into_inner())?;

                    // Apply cached results
                    polars_to_status_error(cache.visit_logical_plan_mut(&mut plan))?;

                    // Actual run
                    let df = run_logical_plan(plan)?;
                    stack.push(ExtendedDataFrame::new(df, Extension::new(stats)));
                }
                CompositePlanSegment::UdfPlanSegment { columns, udf } => {
                    let module =
                        CModule::load_data(&mut Cursor::new(base64::decode(udf).map_err(|e| {
                            Status::invalid_argument(format!(
                                "Could not decode base64-encoded udf: {}",
                                e
                            ))
                        })?))
                        .map_err(|e| {
                            Status::invalid_argument(format!(
                                "Could not deserialize udf from bytes: {}",
                                e
                            ))
                        })?;

                    let mut frame = stack.pop().ok_or_else(|| {
                        Status::invalid_argument("Could not apply udf: no input data frame")
                    })?;
                    for name in columns {
                        let idx = frame
                            .df
                            .get_column_names()
                            .iter()
                            .position(|x| x == &&name)
                            .ok_or_else(|| {
                                Status::invalid_argument(format!(
                                    "Could not apply udf: no column `{}` in data frame",
                                    name
                                ))
                            })?;
                        let series = frame.df.get_columns_mut().get_mut(idx).unwrap();
                        let tensor = series_to_tensor(series)?;
                        let tensor = module.forward_ts(&[tensor]).map_err(|e| {
                            Status::invalid_argument(format!("Error while running udf: {}", e))
                        })?;
                        *series = tensor_to_series(series.name(), series.dtype(), tensor)?;
                    }
                    stack.push(frame);
                }
                CompositePlanSegment::EntryPointPlanSegment(identifier) => {
                    let df = state.get_df_unchecked(&identifier)?;
                    let df = polars_to_status_error(df.with_row_count("__id", None))?;
                    stack.push(ExtendedDataFrame::new(
                        df,
                        Extension::new(PrivacyStats::new(identifier)),
                    ));
                }
                CompositePlanSegment::StackPlanSegment => {
                    let frame1 = stack.pop().ok_or_else(|| {
                        Status::invalid_argument("Could not apply stack: no input data frame")
                    })?;

                    let frame2 = stack.pop().ok_or_else(|| {
                        Status::invalid_argument("Could not apply stack: no df2 input data frame")
                    })?;

                    let df = frame1.df.vstack(&frame2.df).map_err(|e| {
                        Status::invalid_argument(format!("Error while running vstack: {}", e))
                    })?;
                    let mut stats = frame1.extension;

                    polars_to_status_error(
                        stats.privacy_stats.join(frame2.extension.privacy_stats),
                    )?;
                    stack.push(ExtendedDataFrame {
                        df,
                        extension: stats,
                    });
                }
            }
        }

        if stack.len() != 1 {
            return Err(Status::invalid_argument(
                "Wrong number of input data frames",
            ));
        }

        let ExtendedDataFrame { df, extension } = stack.pop().unwrap();
        let Extension { privacy_stats, .. } = extension;
        let mut policy = Policy::allow_by_default();
        let mut blacklist = Vec::new();
        let mut fetchable = VerificationResult::Safe;

        for (identifier, stats) in privacy_stats.0.into_iter() {
            state.with_df_artifact_ref(&identifier, |artifact| -> Result<(), Status> {
                let check = artifact.policy.verify(&Context {
                    stats: stats.clone(),
                    user_id: String::from(user_id),
                    df_identifier: identifier.clone(),
                })?;

                if let VerificationResult::Unsafe { .. } = check {
                    policy = policy.merge(&artifact.policy);
                }
                fetchable.merge(check);
                blacklist.extend_from_slice(&artifact.blacklist[..]);

                Ok(())
            })??;
        }

        Ok(DataFrameArtifact {
            dataframe: df,
            fetchable,
            policy,
            blacklist,
            query_details: plan_str,
        })
    }
}

fn run_logical_plan(plan: LogicalPlan) -> Result<DataFrame, Status> {
    let ldf = lazy_frame_from_logical_plan(plan);
    ldf.collect()
        .map_err(|e| Status::internal(format!("Could not run logical plan: {}", e)))
}
