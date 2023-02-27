use bastionlab_common::common_conversions::{series_to_tensor, tensor_to_series};
use log::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Cursor};
use tch::CModule;

use polars::prelude::*;

use crate::{
    aggregation::plan_pass::{AggPlanCheckPass, AggStats},
    differential_privacy::{DPLogicalPlanAnalyzer, PrivacyStats},
    errors::BastionLabPolarsError,
    init::Initializer,
    state_tree::StateTreeBuilder,
    utils::{CollectDFCacheExt, IntoLazyFrameExt},
    visit::{Visitor, VisitorMut},
};

#[derive(Debug, Clone, PartialEq)]
// not Copy in case we add special arguments to the analyses
pub enum AnalysisMode {
    NoAnalysis,
    DifferentialPrivacy,
    AggregationCheck,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompositePlan {
    pub segments: Vec<CompositePlanSegment>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CompositePlanSegment {
    PolarsPlanSegment { plan: LogicalPlan },
    UdfPlanSegment { columns: Vec<String>, udf: String },
    EntryPointPlanSegment { identifier: String },
    StackPlanSegment,
    RowCountSegment { row: String },
}

pub struct ExtendedLazyFrame<S> {
    pub df: LazyFrame,
    pub extension: S,
}

impl<S> ExtendedLazyFrame<S> {
    pub fn new(df: LazyFrame, extension: S) -> Self {
        ExtendedLazyFrame { df, extension }
    }
}

#[derive(Default)]
pub struct Extension {
    pub privacy_stats: Option<PrivacyStats>,
    pub agg_stats: Option<AggStats>,
}

impl Extension {
    fn new_dp(privacy_stats: PrivacyStats) -> Self {
        Extension {
            privacy_stats: Some(privacy_stats),
            ..Default::default()
        }
    }
    fn new_agg(agg_stats: AggStats) -> Self {
        Extension {
            agg_stats: Some(agg_stats),
            ..Default::default()
        }
    }
    fn empty() -> Self {
        Self::default()
    }

    fn merge(self, other: Self) -> Self {
        Self {
            agg_stats: self
                .agg_stats
                .zip(other.agg_stats)
                .map(|(a, b)| a.merge_least(b)),
            privacy_stats: None,
        }
    }
}

pub struct State(pub HashMap<String, Arc<DataFrame>>);

impl State {
    fn get_df_unchecked(&self, id: &str) -> Result<Arc<DataFrame>, BastionLabPolarsError> {
        Ok(self
            .0
            .get(id)
            .ok_or(BastionLabPolarsError::NotFound(String::from(id)))?
            .clone())
    }
}

impl CompositePlan {
    pub fn get_input_df_ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.segments.iter().filter_map(|e| match e {
            CompositePlanSegment::EntryPointPlanSegment { identifier } => Some(identifier.as_str()),
            _ => None,
        })
    }

    pub fn run(
        self,
        state: State,
        analysis_mode: AnalysisMode,
    ) -> Result<(Arc<DataFrame>, Extension), BastionLabPolarsError> {
        let mut stack = Vec::new();

        for seg in self.segments {
            match seg {
                CompositePlanSegment::PolarsPlanSegment { mut plan } => {
                    // Initialize logical plan
                    let mut init = Initializer::new(&mut stack);
                    init.visit_logical_plan_mut(&mut plan)?;
                    let initial_states = init.initial_states();

                    let lazy_frame = match analysis_mode {
                        AnalysisMode::DifferentialPrivacy => {
                            // Differential Privacy Analysis
                            let dp_initial_states: Vec<_> = initial_states
                                .into_iter()
                                .map(|s: Extension| s.privacy_stats.unwrap())
                                .collect();
                            let mut registry_builder =
                                StateTreeBuilder::from_initial_states(dp_initial_states);
                            registry_builder.visit_logical_plan(&plan)?;
                            let mut cache_builder = StateTreeBuilder::new();
                            cache_builder.visit_logical_plan(&plan)?;
                            let mut dp_analyzer = DPLogicalPlanAnalyzer::new(
                                registry_builder.state_tree()?,
                                cache_builder.state_tree()?,
                            );
                            dp_analyzer.visit_logical_plan(&plan)?;
                            let (stats, mut cache) = dp_analyzer.into_inner()?;

                            // Apply cached results
                            cache.visit_logical_plan_mut(&mut plan)?;

                            ExtendedLazyFrame::new(plan.into_lazy_frame(), Extension::new_dp(stats))
                        }
                        AnalysisMode::AggregationCheck => {
                            // Aggregation check (k-anonymity)
                            let agg_analysis_states: Vec<_> = initial_states
                                .into_iter()
                                .map(|s: Extension| s.agg_stats.unwrap())
                                .collect();
                            let mut registry_builder =
                                StateTreeBuilder::from_initial_states(agg_analysis_states);
                            registry_builder.visit_logical_plan(&plan)?;
                            let mut agg_analyzer =
                                AggPlanCheckPass::new(registry_builder.state_tree()?);
                            agg_analyzer.visit_logical_plan_mut(&mut plan)?;
                            let stats = agg_analyzer.into_inner()?;

                            // Finish running
                            trace!("Finish running! {plan:?}");
                            ExtendedLazyFrame::new(
                                plan.into_lazy_frame(),
                                Extension::new_agg(stats),
                            )
                        }
                        AnalysisMode::NoAnalysis => {
                            ExtendedLazyFrame::new(plan.into_lazy_frame(), Extension::empty())
                        }
                    };

                    // Actual run
                    stack.push(lazy_frame);
                }
                CompositePlanSegment::EntryPointPlanSegment { identifier } => {
                    let df = state.get_df_unchecked(&identifier)?;
                    // let stats = PrivacyStats::try_from_df(&df, identifier.clone())?;
                    let ext = match analysis_mode {
                        AnalysisMode::DifferentialPrivacy => Extension::new_dp(PrivacyStats::new(
                            &identifier,
                            df.clone().into_lazy_frame(),
                        )),
                        AnalysisMode::AggregationCheck => {
                            Extension::new_agg(AggStats::from_input(&df, identifier))
                        }
                        AnalysisMode::NoAnalysis => Extension::empty(),
                    };
                    stack.push(ExtendedLazyFrame::new(df.into_lazy_frame(), ext));
                }
                // todo this should be somewhere else
                CompositePlanSegment::UdfPlanSegment { columns, udf } => {
                    let module =
                        CModule::load_data(&mut Cursor::new(base64::decode(udf).map_err(|e| {
                            BastionLabPolarsError::AnalysisFailed(format!(
                                "Could not decode base64-encoded udf: {}",
                                e
                            ))
                        })?))
                        .map_err(|e| {
                            BastionLabPolarsError::AnalysisFailed(format!(
                                "Could not deserialize udf from bytes: {}",
                                e
                            ))
                        })?;

                    let mut frame = stack.pop().ok_or(BastionLabPolarsError::EmptyStack)?;
                    let mut df = frame.df.collect()?;
                    for name in columns {
                        let idx = df
                            .get_column_names()
                            .iter()
                            .position(|x| x == &&name)
                            .ok_or_else(|| {
                                BastionLabPolarsError::AnalysisFailed(format!(
                                    "Could not apply udf: no column `{}` in data frame",
                                    name
                                ))
                            })?;
                        let series = df.get_columns_mut().get_mut(idx).unwrap();
                        let tensor = series_to_tensor(series).map_err(|e| {
                            BastionLabPolarsError::AnalysisFailed(format!(
                                "Error while running udf: {}",
                                e
                            ))
                        })?;
                        let tensor = module.forward_ts(&[tensor]).map_err(|e| {
                            BastionLabPolarsError::AnalysisFailed(format!(
                                "Error while running udf: {}",
                                e
                            ))
                        })?;
                        *series = tensor_to_series(series.name(), series.dtype(), tensor).map_err(
                            |e| {
                                BastionLabPolarsError::AnalysisFailed(format!(
                                    "Error while running udf: {}",
                                    e
                                ))
                            },
                        )?;
                    }
                    frame.df = df.lazy();
                    stack.push(frame);
                }
                // this should be replaced with LogicalPlan::Union
                // see polars_lazy::dsl::functions::concat(&[impl AsRef<LazyFrame>]) -> LazyFrame
                CompositePlanSegment::StackPlanSegment => {
                    let frame1 = stack.pop().ok_or(BastionLabPolarsError::EmptyStack)?;
                    let frame2 = stack.pop().ok_or(BastionLabPolarsError::EmptyStack)?;

                    let df1 = frame1.df.collect_or_get_cached()?;
                    let df2 = frame2.df.collect_or_get_cached()?;

                    let df = df1
                        .vstack(&df2)
                        .map_err(|e| BastionLabPolarsError::LogicalPlanErr(e))?
                        .lazy();
                    stack.push(ExtendedLazyFrame::new(
                        df,
                        frame1.extension.merge(frame2.extension),
                    ));
                }
                CompositePlanSegment::RowCountSegment { row: name } => {
                    let frame = stack.pop().ok_or(BastionLabPolarsError::EmptyStack)?;
                    let df = frame.df.with_row_count(&name, Some(0));
                    stack.push(ExtendedLazyFrame::new(df, frame.extension));
                }
            }
        }

        let el = stack.pop().ok_or(BastionLabPolarsError::EmptyStack)?;
        // Finish running
        let df = el.df.collect_or_get_cached()?;
        Ok((df, el.extension))
    }
}
