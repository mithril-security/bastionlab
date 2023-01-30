use std::collections::HashMap;

use polars::prelude::*;

use crate::{
    cache::Cache,
    errors::BastionLabPolarsError,
    state_tree::{StateNodeChildren, StateTree},
    visit::{self, Visitor},
};

pub fn lazy_frame_from_logical_plan(plan: LogicalPlan) -> LazyFrame {
    let mut ldf = LazyFrame::default();
    ldf.logical_plan = plan;
    ldf
}

fn usize_item(df_res: Result<DataFrame, PolarsError>) -> Result<usize, PolarsError> {
    Ok(df_res?.get(0).unwrap()[0].try_extract().unwrap())
}

/// Encodes the sensibility of a function evaluation w.r.t. its input expressed
/// either in `LInfinity` or `L2` norm. The `Unbounded` variant is used
/// when the sensibility cannot be automatically infered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LInfinitySensibility {
    Unbounded,
    Bounded(f32),
}

impl LInfinitySensibility {
    pub fn max(self, other: Self) -> Self {
        match (self, other) {
            (LInfinitySensibility::Bounded(a), LInfinitySensibility::Bounded(b)) => {
                LInfinitySensibility::Bounded(a.max(b))
            }
            _ => LInfinitySensibility::Unbounded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StructuralDependence {
    Unbounded,
    MaxRepeat(usize),
}

impl StructuralDependence {
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (StructuralDependence::MaxRepeat(a), StructuralDependence::MaxRepeat(b)) => {
                StructuralDependence::MaxRepeat(a.max(b))
            }
            _ => StructuralDependence::Unbounded,
        }
    }
}

/// Represents a privacy budget (epsilon) that is possibly infinite (`NotPrivate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrivacyBudget {
    NotPrivate,
    Private(f32),
}

/// Contains all the necessary data for [`Dataset`] to track its usage.
///
/// This struct is placed in an [`Arc`] and shared with all PrivacyGuards
/// that contain data from the dataset so that the guards can increase the
/// expended budget when turned into readable values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrivacyContext {
    expended: PrivacyBudget,
    limit: PrivacyBudget,
    delta: f32,
    nb_samples: usize,
}

impl PrivacyContext {
    pub fn new(limit: PrivacyBudget, nb_samples: usize) -> Self {
        PrivacyContext {
            expended: PrivacyBudget::Private(0.0),
            limit,
            delta: 1.0 / (10.0 * nb_samples as f32),
            nb_samples,
        }
    }

    pub fn within_bounds(&self, budget: PrivacyBudget) -> bool {
        match &self.limit {
            PrivacyBudget::NotPrivate => true,
            PrivacyBudget::Private(eps_limit) => match &self.expended {
                PrivacyBudget::NotPrivate => false,
                PrivacyBudget::Private(eps_expended) => match budget {
                    PrivacyBudget::NotPrivate => false,
                    PrivacyBudget::Private(eps) => eps + eps_expended < *eps_limit,
                },
            },
        }
    }

    pub fn delta(&self) -> f32 {
        self.delta
    }

    pub fn nb_samples(&self) -> usize {
        self.nb_samples
    }

    fn update_budget(&mut self, budget: PrivacyBudget) {
        match (&mut self.expended, budget) {
            (PrivacyBudget::NotPrivate, _) => (),
            (PrivacyBudget::Private(_), PrivacyBudget::NotPrivate) => {
                self.expended = PrivacyBudget::NotPrivate
            }
            (PrivacyBudget::Private(eps_expended), PrivacyBudget::Private(eps)) => {
                *eps_expended += eps;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrivacyStatsEntry {
    sensibility: LInfinitySensibility,
    structural_dependence: StructuralDependence,
}

impl PrivacyStatsEntry {
    fn new() -> Self {
        PrivacyStatsEntry {
            sensibility: LInfinitySensibility::Unbounded,
            structural_dependence: StructuralDependence::MaxRepeat(1),
        }
    }

    fn update_sensibility_inplace(&mut self, sensibility: f32) {
        match &mut self.sensibility {
            LInfinitySensibility::Unbounded => (),
            LInfinitySensibility::Bounded(bound) => *bound *= sensibility,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrivacyStats(pub(crate) HashMap<String, PrivacyStatsEntry>);

impl PrivacyStats {
    pub fn new(identifier: String) -> Self {
        let mut res = HashMap::new();
        res.insert(identifier, PrivacyStatsEntry::new());
        PrivacyStats(res)
    }

    fn update_sensibility_inplace(&mut self, sensibility: f32) {
        for entry in self.0.values_mut() {
            entry.update_sensibility_inplace(sensibility)
        }
    }

    fn update_sensibility(&self, sensibility: f32) -> Self {
        let mut res = self.clone();
        res.update_sensibility_inplace(sensibility);
        res
    }

    pub fn join(&mut self, other: Self) -> Result<(), BastionLabPolarsError> {
        for (identifier, entry_left) in self.0.iter_mut() {
            if let Some(entry_right) = other.0.get(identifier) {
                entry_left.sensibility = entry_left.sensibility.max(entry_right.sensibility);
                entry_left.structural_dependence = entry_left
                    .structural_dependence
                    .and(entry_right.structural_dependence);
            }
        }

        for (identifier, entry_right) in other.0.into_iter() {
            if let None = self.0.get(&identifier) {
                self.0.insert(identifier, entry_right);
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct DPAnalyzerBasePass {
    registry: StateTree<PrivacyStats>,
    cache: Cache,
}

impl Visitor for DPAnalyzerBasePass {
    fn visit_logical_plan(
        &mut self,
        node: &LogicalPlan,
    ) -> Result<(), crate::errors::BastionLabPolarsError> {
        self.registry.next();
        self.cache.0.next();
        visit::visit_logical_plan(self, node)?;

        match node {
            LogicalPlan::Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let StateNodeChildren::Binary(left_node, right_node) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };

                let left_ldf = lazy_frame_from_logical_plan((&**input_left).clone());
                let right_ldf = lazy_frame_from_logical_plan((&**input_right).clone());

                let (left_ldf, right_ldf) = self.cache.cache_binary(left_ldf, right_ldf)?;

                let left_ldf = left_ldf.with_row_count("__ids", None);
                let right_ldf = right_ldf.with_row_count("__ids", None);

                match options.how {
                    JoinType::Anti | JoinType::Semi => {
                        *self.registry.state.borrow_mut() = left_node.state.borrow().clone()
                    }
                    _ => {
                        let joined_ids = left_ldf
                            .cache()
                            .join(right_ldf.cache(), left_on, right_on, options.how.clone())
                            .select([col("__ids"), col("__ids_right")])
                            .cache();

                        let left_join_scaling = usize_item(
                            joined_ids
                                .clone()
                                .groupby([col("__ids")])
                                .agg([col("__ids_right").count()])
                                .select([col("__ids_right").max()])
                                .collect(),
                        )
                        .map_err(|_| BastionLabPolarsError::LeftJoinScalingErr)?;

                        let right_join_scaling = usize_item(
                            joined_ids
                                .groupby([col("__ids_right")])
                                .agg([col("__ids").count()])
                                .select([col("__ids").max()])
                                .collect(),
                        )
                        .map_err(|_| BastionLabPolarsError::RightJoinScalingErr)?;

                        let mut left_stats = left_node
                            .state
                            .borrow()
                            .as_ref()
                            .ok_or(BastionLabPolarsError::BadState)?
                            .update_sensibility(left_join_scaling as f32);
                        let right_stats = right_node
                            .state
                            .borrow()
                            .as_ref()
                            .ok_or(BastionLabPolarsError::BadState)?
                            .update_sensibility(right_join_scaling as f32);
                        left_stats.join(right_stats)?;
                        *self.registry.state.borrow_mut() = Some(left_stats);
                    }
                }
            }
            // These are not currently supported
            // LogicalPlan::ExtContext { .. } => *state = false,
            // LogicalPlan::Union { .. } => *state = false,

            // LogicalPlan::Projection { expr, .. } => {}
            // LogicalPlan::LocalProjection { expr, .. } => {}
            // LogicalPlan::Aggregate {
            //     input, keys, aggs, ..
            // } => {}
            _ => (),
        }

        self.registry.prev();
        self.cache.0.prev();

        Ok(())
    }

    fn visit_expr(&mut self, node: &Expr) -> Result<(), BastionLabPolarsError> {
        visit::visit_expr(self, node)?;
        println!("Visiting expression");
        // match node {
        //     Expr::Column(_) | Expr::Columns(_) => self.expr_stack.push(self.main_stack.last().ok_or(BastionLabPolarsError::EmptyStack)?.clone()),
        //     Expr::DtypeColumn(_) => self.expr_stack.push(PrivacyStats::new()),
        //     _ => (),
        // }

        Ok(())
    }
}

impl DPAnalyzerBasePass {
    pub fn new(registry: StateTree<PrivacyStats>, cache: StateTree<LazyFrame>) -> Self {
        DPAnalyzerBasePass {
            registry,
            cache: Cache(cache),
        }
    }

    pub fn into_inner(self) -> Result<(PrivacyStats, Cache), BastionLabPolarsError> {
        let node = self.registry.try_unwrap()?;

        let stats = node
            .state
            .into_inner()
            .ok_or(BastionLabPolarsError::BadState)?;
        Ok((stats, self.cache))
    }

    // fn analyze(&mut self, plan: &LogicalPlan) -> Result<(), PolarsError> {
    //     match plan {
    //         LogicalPlan::Join {
    //             input_left,
    //             input_right,
    //             left_on,
    //             right_on,
    //             options,
    //             ..
    //         } => {
    //             let left_stats = PrivacyStat

    //             let left_ldf = lazy_frame_from_logical_plan((&**input_left).clone()).with_row_count("__ids", None);
    //             let right_ldf = lazy_frame_from_logical_plan((&**input_right).clone()).with_row_count("__ids", None);

    //             match options.how {
    //                 JoinType::Anti | JoinType::Semi => right_input.as_mut().unwrap().reset(),
    //                 _ => {
    //                     let joined_ids = left_ldf
    //                         .cache()
    //                         .join(
    //                             right_ldf.cache(),
    //                             left_on,
    //                             right_on,
    //                             options.how.clone(),
    //                         )
    //                         .select([col("__ids"), col("__ids_right")])
    //                         .cache();

    //                     let left_join_scaling = usize_item(
    //                         joined_ids
    //                             .clone()
    //                             .groupby([col("__ids")])
    //                             .agg([col("__ids_right").count()])
    //                             .select([col("__ids_right").max()])
    //                             .collect(),
    //                     )?;

    //                     let right_join_scaling = usize_item(
    //                         joined_ids
    //                             .groupby([col("__ids_right")])
    //                             .agg([col("__ids").count()])
    //                             .select([col("__ids").max()])
    //                             .collect(),
    //                     )?;

    //                     left_input.update_sensibility(left_join_scaling as f32);
    //                     right_input.as_mut().unwrap().update_sensibility(right_join_scaling as f32);

    //                 }
    //             }

    //             left.merge(right);
    //             stats_stack.push(left);
    //         }
    //         // These are not currently supported
    //         // LogicalPlan::ExtContext { .. } => *state = false,
    //         // LogicalPlan::Union { .. } => *state = false,
    //         LogicalPlan::Projection { expr, .. } => {
    //             if exprs_agg_check(expr)? {
    //                 stats_stack.last_mut().unwrap().update_agg_size(usize::MAX);
    //             }
    //         }
    //         LogicalPlan::LocalProjection { expr, .. } => {
    //             if exprs_agg_check(expr)? {
    //                 stats_stack.last_mut().unwrap().update_agg_size(usize::MAX);
    //             }
    //         }
    //         LogicalPlan::Aggregate {
    //             input, keys, aggs, ..
    //         } => {
    //             let keys = &(**keys)[..];
    //             let ldf = lazy_frame_from_logical_plan((&**input).clone());
    //             let agg_size = usize_item(
    //                 ldf.cache()
    //                     .with_row_count("__count", None)
    //                     .groupby(keys)
    //                     .agg([col("__count").count()])
    //                     .select([col("__count").min()])
    //                     .collect(),
    //             )?;

    //             if exprs_agg_check(aggs)? {
    //                 stats_stack.last_mut().unwrap().update_agg_size(agg_size);
    //             }
    //         }
    //         _ => (),
    //     }

    //     PrivacyStats {
    //         sensibility: Sensibility::Unbounded,
    //         structural_dependence: false,
    //         dependencies: vec![],
    //     }
    // }
}
