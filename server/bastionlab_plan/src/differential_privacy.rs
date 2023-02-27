use std::collections::HashMap;

use polars::{export::num::Signed, lazy::dsl::Operator, prelude::*};

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

#[derive(Clone)]
pub struct PrivacyStats {
    sensibility: LazyFrame,
    independent_lines: bool,
    columns_origins: HashMap<String, String>,
}

impl PrivacyStats {
    pub fn new(identifier: &str, df: LazyFrame) -> Self {
        let cols: Vec<String> = df.schema().unwrap().iter_names().cloned().collect(); // FIX THIS
        let mut columns_origins = HashMap::new();
        for c in cols {
            columns_origins.insert(c, String::from(identifier));
        }
        PrivacyStats {
            sensibility: df.select([col("*").abs()]),
            independent_lines: true,
            columns_origins,
        }
    }
}

#[derive(Clone)]
pub struct DPExprAnalyzer {
    stack: Vec<Expr>,
    named_inputs: Vec<String>,
    wildcard: bool,
}

impl Visitor for DPExprAnalyzer {
    fn visit_expr(&mut self, node: &Expr) -> Result<(), BastionLabPolarsError> {
        visit::visit_expr(self, node)?;

        match node {
            Expr::Column(name) => {
                self.named_inputs.push(String::from(&**name));
                self.stack.push(node.clone().cast(DataType::Float64).abs());
            }
            Expr::Columns(names) => {
                self.named_inputs.extend_from_slice(&names);
                self.stack.push(node.clone().cast(DataType::Float64).abs());
            }
            Expr::DtypeColumn(_) => {
                self.wildcard = true;
                self.stack.push(node.clone().cast(DataType::Float64).abs());
            }
            Expr::Literal(_) => {
                self.stack.push(lit(0.0));
            }
            Expr::BinaryExpr { op, .. } => {
                let right = self.stack.pop().ok_or(BastionLabPolarsError::BadState)?;
                let left = self.stack.pop().ok_or(BastionLabPolarsError::BadState)?;

                let res = match op {
                    Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
                    | Operator::And
                    | Operator::Or
                    | Operator::Xor => left.gt(0).or(right.gt(0)).cast(DataType::Float64),
                    Operator::Plus | Operator::Minus => left + right,
                    // Operator::Multiply
                    _ => left,
                };
            }
            _ => (),
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct DPLogicalPlanAnalyzer {
    registry: StateTree<PrivacyStats>,
    cache: Cache,
}

impl Visitor for DPLogicalPlanAnalyzer {
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

                        // let left_stats = left_node
                        //     .state
                        //     .borrow()
                        //     .as_ref()
                        //     .ok_or(BastionLabPolarsError::BadState)?
                        //     .update_sensibility(left_join_scaling as f32);
                        // let right_stats = right_node
                        //     .state
                        //     .borrow()
                        //     .as_ref()
                        //     .ok_or(BastionLabPolarsError::BadState)?
                        //     .update_sensibility(right_join_scaling as f32);
                        // let stats = left_stats.join(right_stats)?;
                        // *self.registry.state.borrow_mut() = Some(stats);
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
            LogicalPlan::DataFrameScan { .. } => (),
            _ => {
                let StateNodeChildren::Unary(node) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };
                *self.registry.state.borrow_mut() = node.state.borrow().clone();
            }
        }

        self.registry.prev();
        self.cache.0.prev();

        Ok(())
    }

    // Do not recurse on Expr and AggExpr
    fn visit_expr(&mut self, _node: &Expr) -> Result<(), BastionLabPolarsError> {
        Ok(())
    }
    fn visit_agg_expr(&mut self, _node: &AggExpr) -> Result<(), BastionLabPolarsError> {
        Ok(())
    }
}

impl DPLogicalPlanAnalyzer {
    pub fn new(registry: StateTree<PrivacyStats>, cache: StateTree<LazyFrame>) -> Self {
        DPLogicalPlanAnalyzer {
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
