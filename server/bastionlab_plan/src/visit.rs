use crate::errors::BastionLabPolarsError;
use polars::lazy::dsl::{AggExpr, Expr};
use polars::prelude::*;

macro_rules! visitor_trait {
    ($($visitor_trait:ident  { $($mut:tt)? } { $iter:ident } { $($arc_mut:path)? } {
        $visit_logical_plan:ident
        $visit_expr:ident
        $visit_agg_expr:ident
    })*) => {$(
        pub trait $visitor_trait {
            fn $visit_logical_plan(&mut self, node: &$($mut)? LogicalPlan) -> Result<(), BastionLabPolarsError> {
                $visit_logical_plan(self, node)
            }

            fn $visit_expr(&mut self, node: &$($mut)? Expr) -> Result<(), BastionLabPolarsError> {
                $visit_expr(self, node)
            }

            fn $visit_agg_expr(&mut self, node: &$($mut)? AggExpr) -> Result<(), BastionLabPolarsError> {
                $visit_agg_expr(self, node)
            }
        }

        pub fn $visit_logical_plan<V: $visitor_trait + ?Sized>(v: &mut V, node: &$($mut)? LogicalPlan) -> Result<(), BastionLabPolarsError> {
            match node {
                LogicalPlan::DataFrameScan { .. } => (),
                LogicalPlan::Distinct { input, .. }
                | LogicalPlan::Explode { input, .. }
                | LogicalPlan::Slice { input, .. }
                | LogicalPlan::Melt { input, .. }
                | LogicalPlan::MapFunction { input, .. } => v.$visit_logical_plan(input)?,
                LogicalPlan::LocalProjection { expr, input, .. }
                | LogicalPlan::Projection { expr, input, .. } => {
                    v.$visit_logical_plan(input)?;
                    for e in expr.$iter() {
                        v.$visit_expr(e)?
                    }
                },
                LogicalPlan::Aggregate { input, keys, aggs, .. } => {
                    v.$visit_logical_plan(input)?;
                    $(let keys = $arc_mut(keys);)?
                    for e in keys.$iter() {
                        v.$visit_expr(e)?
                    }
                    for e in aggs.$iter() {
                        v.$visit_expr(e)?
                    }
                }
                LogicalPlan::Join {
                    input_left,
                    input_right,
                    left_on,
                    right_on,
                    ..
                } => {
                    v.$visit_logical_plan(input_left)?;
                    v.$visit_logical_plan(input_right)?;
                    for e in left_on.$iter() {
                        v.$visit_expr(e)?
                    }
                    for e in right_on.$iter() {
                        v.$visit_expr(e)?
                    }
                }
                LogicalPlan::Sort { input, by_column, .. } => {
                    v.$visit_logical_plan(input)?;
                    for e in by_column.$iter() {
                        v.$visit_expr(e)?
                    }
                }
                LogicalPlan::Selection { input, predicate } => {
                    v.$visit_logical_plan(input)?;
                    v.$visit_expr(predicate)?;
                }
                LogicalPlan::HStack { input, exprs, .. } => {
                    v.$visit_logical_plan(input)?;
                    for e in exprs.$iter() {
                        v.$visit_expr(e)?
                    }
                }
                LogicalPlan::Error { input, err, .. } => {
                    v.$visit_logical_plan(input)?;
                    ::log::trace!("error in plan: {err:?}");
                    // return Err(BastionLabPolarsError::LogicalPlanErr(node.schema().err().unwrap()))?
                }
                _ => {
                    ::log::trace!("unsupported plan variant: {node:?}");
                    return Err(BastionLabPolarsError::UnsupportedLogicalPlanVariant)
                }
            }

            Ok(())
        }

        pub fn $visit_expr<V: $visitor_trait + ?Sized>(v: &mut V, node: &$($mut)? Expr) -> Result<(), BastionLabPolarsError> {
            match node {
                Expr::Alias(nested, _) => v.$visit_expr(nested)?,
                Expr::Column(_) => (),
                Expr::Columns(_) => (),
                Expr::DtypeColumn(_) => (),
                Expr::Literal(_) => (),
                Expr::BinaryExpr { left, right, .. } => {
                    v.$visit_expr(left)?;
                    v.$visit_expr(right)?;
                }
                Expr::Cast { expr, .. } => v.$visit_expr(expr)?,
                Expr::Sort { expr, .. } => v.$visit_expr(expr)?,
                Expr::Take { expr, idx } => {
                    v.$visit_expr(expr)?;
                    v.$visit_expr(idx)?;
                }
                Expr::SortBy { expr, by, .. } => {
                    v.$visit_expr(expr)?;
                    for e in by {
                        v.$visit_expr(e)?;
                    }
                }
                Expr::Agg(a) => v.$visit_agg_expr(a)?,
                Expr::Ternary { predicate, truthy, falsy } => {
                    v.$visit_expr(predicate)?;
                    v.$visit_expr(truthy)?;
                    v.$visit_expr(falsy)?;
                }
                Expr::Function { input, .. } => {
                    for e in input {
                        v.$visit_expr(e)?;
                    }
                }
                Expr::Explode(input) => v.$visit_expr(input)?,
                Expr::Filter { input, by } => {
                    v.$visit_expr(input)?;
                    v.$visit_expr(by)?;
                }
                Expr::Window { function, partition_by, /*order_by,*/ .. } => {
                    v.$visit_expr(function)?;
                    for e in partition_by {
                        v.$visit_expr(e)?;
                    }
                    // if let Some(e) = order_by {
                    //     e.$visitable_fn(state, f.clone())?;
                    // }
                }
                Expr::Wildcard => (),
                Expr::Slice { input, offset, length } => {
                    v.$visit_expr(input)?;
                    v.$visit_expr(offset)?;
                    v.$visit_expr(length)?;
                }
                Expr::Exclude(input, _) => v.$visit_expr(input)?,
                Expr::KeepName(input) => v.$visit_expr(input)?,
                Expr::Count => (),
                Expr::Nth(_) => (),
                _ => {
                    return Err(BastionLabPolarsError::UnsupportedExprVariant)
                }
            }

            Ok(())
        }

        pub fn $visit_agg_expr<V: $visitor_trait + ?Sized>(v: &mut V, node: &$($mut)? AggExpr) -> Result<(), BastionLabPolarsError> {
            match node {
                AggExpr::Min { input, .. } => v.$visit_expr(input)?,
                AggExpr::Max { input, .. } => v.$visit_expr(input)?,
                AggExpr::Median(e) => v.$visit_expr(e)?,
                AggExpr::NUnique(e) => v.$visit_expr(e)?,
                AggExpr::First(e) => v.$visit_expr(e)?,
                AggExpr::Last(e) => v.$visit_expr(e)?,
                AggExpr::Mean(e) => v.$visit_expr(e)?,
                AggExpr::List(e) => v.$visit_expr(e)?,
                AggExpr::Count(e) => v.$visit_expr(e)?,
                AggExpr::Quantile { expr, .. } => v.$visit_expr(expr)?,
                AggExpr::Sum(e) => v.$visit_expr(e)?,
                AggExpr::AggGroups(e) => v.$visit_expr(e)?,
                AggExpr::Std(e, _) => v.$visit_expr(e)?,
                AggExpr::Var(e, _) => v.$visit_expr(e)?,
            }

            Ok(())
        }
    )*};
}

visitor_trait! {
    Visitor {} { iter } {} {
        visit_logical_plan
        visit_expr
        visit_agg_expr
    }
    VisitorMut { mut } { iter_mut } { Arc::make_mut } {
        visit_logical_plan_mut
        visit_expr_mut
        visit_agg_expr_mut
    }
}
