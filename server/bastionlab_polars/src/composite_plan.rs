use base64;
use polars::{lazy::dsl::Expr, prelude::*};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Cursor};
use tch::CModule;
use tonic::Status;

use crate::{
    access_control::{Context, Policy, VerificationResult},
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

#[derive(Debug, Clone, Copy)]
pub struct StatsEntry {
    pub agg_size: usize,
    pub join_scaling: usize,
}

#[derive(Debug, Clone)]
pub struct DataFrameStats(HashMap<String, StatsEntry>);

struct StackFrame {
    df: DataFrame,
    stats: DataFrameStats,
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
                    let stats = initialize_plan(&mut plan, &mut stack)?;
                    let df = run_logical_plan(plan)?;
                    stack.push(StackFrame { df, stats });
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
                    let stats = DataFrameStats::new(identifier);
                    stack.push(StackFrame { df, stats });
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
                    let mut stats = frame1.stats;
                    stats.merge(frame2.stats);
                    stack.push(StackFrame { df, stats });
                }
            }
        }

        if stack.len() != 1 {
            return Err(Status::invalid_argument(
                "Wrong number of input data frames",
            ));
        }

        let StackFrame { df, stats } = stack.pop().unwrap();

        let mut policy = Policy::allow_by_default();
        let mut blacklist = Vec::new();
        let mut fetchable = VerificationResult::Safe;

        for (identifier, stats) in stats.0.into_iter() {
            state.with_df_artifact_ref(&identifier, |artifact| -> Result<(), Status> {
                let check = artifact.policy.verify(&Context {
                    stats,
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

fn expr_agg_check(expr: &Expr) -> Result<bool, Status> {
    let mut state = Vec::new();
    expr.visit(&mut state, |expr, state| {
        match expr {
            Expr::Column(_)
            | Expr::Columns(_)
            | Expr::DtypeColumn(_)
            | Expr::Wildcard
            | Expr::Nth(_) => state.push(false),
            Expr::Literal(_) | Expr::Count => state.push(true),
            Expr::Agg(AggExpr::List(expr)) | Expr::Agg(AggExpr::AggGroups(expr)) => {
                state.push(expr_agg_check(&expr)?)
            }
            Expr::Agg(_) => state.push(true),
            Expr::BinaryExpr { .. } => {
                let right_agg = state.pop().unwrap();
                let left_agg = state.pop().unwrap();
                state.push(right_agg && left_agg);
            }
            Expr::Take { .. } | Expr::SortBy { .. } | Expr::Filter { .. } | Expr::Window { .. } => {
                state.pop().unwrap();
                let expr_agg = state.pop().unwrap();
                state.push(expr_agg);
            }
            Expr::Ternary { .. } => {
                let falsy_agg = state.pop().unwrap();
                let truthy_agg = state.pop().unwrap();
                state.pop().unwrap();
                state.push(truthy_agg && falsy_agg);
            }
            Expr::Slice { .. } => {
                state.pop().unwrap();
                state.pop().unwrap();
                let input_agg = state.pop().unwrap();
                state.push(input_agg);
            }
            _ => (),
        }

        Ok(())
    })?;

    Ok(state.pop().unwrap())
}

fn exprs_agg_check(exprs: &[Expr]) -> Result<bool, Status> {
    for e in exprs.iter() {
        let x = expr_agg_check(e)?;
        if !x {
            return Ok(false);
        }
    }
    Ok(true)
}

fn run_logical_plan(plan: LogicalPlan) -> Result<DataFrame, Status> {
    let ldf = lazy_frame_from_logical_plan(plan);
    ldf.collect()
        .map_err(|e| Status::internal(format!("Could not run logical plan: {}", e)))
}

impl DataFrameStats {
    fn new(identifier: String) -> DataFrameStats {
        let mut stats = HashMap::new();
        stats.insert(
            identifier,
            StatsEntry {
                agg_size: 1,
                join_scaling: 1,
            },
        );
        DataFrameStats(stats)
    }

    fn update_agg_size(&mut self, agg_size: usize) {
        for stats in self.0.values_mut() {
            stats.agg_size *= agg_size;
        }
    }

    fn update_join_scaling(&mut self, join_scaling: usize) {
        for stats in self.0.values_mut() {
            stats.join_scaling *= join_scaling;
        }
    }

    fn merge(&mut self, other: DataFrameStats) {
        for (identifier, stats_left) in self.0.iter_mut() {
            if let Some(stats_right) = other.0.get(identifier) {
                stats_left.agg_size = stats_left.agg_size.min(stats_right.agg_size);
                stats_left.join_scaling = stats_left.join_scaling.max(stats_right.join_scaling);
            }
        }

        for (identifier, stats_right) in other.0.into_iter() {
            if let None = self.0.get(&identifier) {
                self.0.insert(identifier, stats_right);
            }
        }
    }
}

fn initialize_plan(
    plan: &mut LogicalPlan,
    stack: &mut Vec<StackFrame>,
) -> Result<DataFrameStats, Status> {
    let mut state = (stack, Vec::new());
    plan.visit_mut(&mut state, |plan, (main_stack, stats_stack)| {
        match plan {
            LogicalPlan::DataFrameScan { .. } => {
                let frame = main_stack.pop().ok_or_else(|| {
                    Status::invalid_argument(
                        "Could not run logical plan: not enough input data frames",
                    )
                })?;
                stats_stack.push(frame.stats);
                *plan = frame.df.lazy().logical_plan;
            }
            LogicalPlan::Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let left_ldf = lazy_frame_from_logical_plan((&**input_left).clone());
                let right_ldf = lazy_frame_from_logical_plan((&**input_right).clone());

                let mut right = stats_stack.pop().unwrap();
                let mut left = stats_stack.pop().unwrap();

                match options.how {
                    JoinType::Anti | JoinType::Semi => right.update_join_scaling(0),
                    _ => {
                        let joined_ids = left_ldf
                            .cache()
                            .with_row_count("__left_count", None)
                            .join(
                                right_ldf.cache().with_row_count("__right_count", None),
                                left_on,
                                right_on,
                                options.how.clone(),
                            )
                            .select([col("__left_count"), col("__right_count")])
                            .cache();

                        let left_join_scaling = usize_item(
                            joined_ids
                                .clone()
                                .groupby([col("__left_count")])
                                .agg([col("__right_count").count()])
                                .select([col("__right_count").max()])
                                .collect(),
                        )?;

                        let right_join_scaling = usize_item(
                            joined_ids
                                .groupby([col("__right_count")])
                                .agg([col("__left_count").count()])
                                .select([col("__left_count").max()])
                                .collect(),
                        )?;

                        right.update_join_scaling(right_join_scaling);
                        left.update_join_scaling(left_join_scaling);
                    }
                }

                left.merge(right);
                stats_stack.push(left);
            }
            // These are not currently supported
            // LogicalPlan::ExtContext { .. } => *state = false,
            // LogicalPlan::Union { .. } => *state = false,
            LogicalPlan::Projection { expr, .. } => {
                if exprs_agg_check(expr)? {
                    stats_stack.last_mut().unwrap().update_agg_size(usize::MAX);
                }
            }
            LogicalPlan::LocalProjection { expr, .. } => {
                if exprs_agg_check(expr)? {
                    stats_stack.last_mut().unwrap().update_agg_size(usize::MAX);
                }
            }
            LogicalPlan::Aggregate {
                input, keys, aggs, ..
            } => {
                let keys = &(**keys)[..];
                let ldf = lazy_frame_from_logical_plan((&**input).clone());
                let agg_size = usize_item(
                    ldf.cache()
                        .with_row_count("__count", None)
                        .groupby(keys)
                        .agg([col("__count").count()])
                        .select([col("__count").min()])
                        .collect(),
                )?;

                if exprs_agg_check(aggs)? {
                    stats_stack.last_mut().unwrap().update_agg_size(agg_size);
                }
            }
            _ => (),
        }
        Ok(())
    })?;

    Ok(state.1.pop().unwrap())
}

fn usize_item(df_res: Result<DataFrame, PolarsError>) -> Result<usize, Status> {
    Ok(df_res
        .map_err(|e| Status::internal(format!("Could not get usize item from DataFrame: {}", e)))?
        .get(0)
        .unwrap()[0]
        .try_extract()
        .unwrap())
}
