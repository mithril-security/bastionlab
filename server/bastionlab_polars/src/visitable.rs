use polars::prelude::{Expr, LogicalPlan};
use tonic::Status;

pub trait Visitable<S> {
    fn visit<F: Fn(&Self, &mut S) -> Result<(), Status> + Clone>(
        &self,
        state: &mut S,
        f: F,
    ) -> Result<(), Status>;
}

pub trait VisitableMut<S> {
    fn visit_mut<F: Fn(&mut Self, &mut S) -> Result<(), Status> + Clone>(
        &mut self,
        state: &mut S,
        f: F,
    ) -> Result<(), Status>;
}

macro_rules! visitable_impl_logical_plan {
    ($($visitable_trait:ident $visitable_fn:ident $self_ty:ty)*) => {$(
        impl<S> $visitable_trait<S> for LogicalPlan {
            fn $visitable_fn<F: Fn($self_ty, &mut S) -> Result<(), Status> + Clone>(self: $self_ty, state: &mut S, f: F) -> Result<(), Status> {
                match self {
                    LogicalPlan::DataFrameScan { .. } => (),
                    LogicalPlan::Selection { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Cache { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::LocalProjection { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Projection { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Aggregate { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Join {
                        input_left,
                        input_right,
                        ..
                    } => {
                        input_left.$visitable_fn(state, f.clone())?;
                        input_right.$visitable_fn(state, f.clone())?;
                    }
                    LogicalPlan::HStack { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Distinct { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Sort { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Explode { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Slice { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::Melt { input, .. } => input.$visitable_fn(state, f.clone())?,
                    LogicalPlan::MapFunction { input, .. } => input.$visitable_fn(state, f.clone())?,
                    // We currently don't need support for this
                    // LogicalPlan::Union { inputs, .. } => {
                    //     for input in inputs {
                    //         input.$visitable_fn(state, f.clone())?;
                    //     }
                    // }
                    // LogicalPlan::ExtContext {
                    //     input, contexts, ..
                    // } => {
                    //     input.$visitable_fn(state, f.clone())?;
                    //     for context in contexts {
                    //         context.$visitable_fn(state, f.clone())?;
                    //     }
                    // }
                    lp => {
                        return Err(Status::invalid_argument(format!(
                            "Logical plan contains unsupported variants: {:?}",
                            lp
                        )))
                    }
                }
                f(self, state)?;
                Ok(())
            }
        }

    )*};
}

visitable_impl_logical_plan! {
    Visitable visit &Self
    VisitableMut visit_mut &mut Self
}

macro_rules! visitable_impl_expr {
    ($($visitable_trait:ident $visitable_fn:ident $self_ty:ty)*) => {$(
        impl<S> $visitable_trait<S> for Expr {
            fn $visitable_fn<F: Fn($self_ty, &mut S) -> Result<(), Status> + Clone>(self: $self_ty, state: &mut S, f: F) -> Result<(), Status> {
                match self {
                    Expr::Alias(nested, _) => nested.$visitable_fn(state, f.clone())?,
                    Expr::Column(_) => (),
                    Expr::Columns(_) => (),
                    Expr::DtypeColumn(_) => (),
                    Expr::Literal(_) => (),
                    Expr::BinaryExpr { left, right, .. } => {
                        left.$visitable_fn(state, f.clone())?;
                        right.$visitable_fn(state, f.clone())?;
                    }
                    Expr::Cast { expr, .. } => expr.$visitable_fn(state, f.clone())?,
                    Expr::Sort { expr, .. } => expr.$visitable_fn(state, f.clone())?,
                    Expr::Take { expr, idx } => {
                        expr.$visitable_fn(state, f.clone())?;
                        idx.$visitable_fn(state, f.clone())?;
                    }
                    Expr::SortBy { expr, by, .. } => {
                        expr.$visitable_fn(state, f.clone())?;
                        for e in by {
                            e.$visitable_fn(state, f.clone())?;
                        }
                    }
                    Expr::Agg(_) => (),
                    Expr::Ternary { predicate, truthy, falsy } => {
                        predicate.$visitable_fn(state, f.clone())?;
                        truthy.$visitable_fn(state, f.clone())?;
                        falsy.$visitable_fn(state, f.clone())?;
                    }
                    Expr::Function { input, .. } => {
                        for e in input {
                            e.$visitable_fn(state, f.clone())?;
                        }
                    }
                    Expr::Explode(input) => input.$visitable_fn(state, f.clone())?,
                    Expr::Filter { input, by } => {
                        input.$visitable_fn(state, f.clone())?;
                        by.$visitable_fn(state, f.clone())?;
                    }
                    Expr::Window { function, partition_by, /*order_by,*/ .. } => {
                        function.$visitable_fn(state, f.clone())?;
                        for e in partition_by {
                            e.$visitable_fn(state, f.clone())?;
                        }
                        // if let Some(e) = order_by {
                        //     e.$visitable_fn(state, f.clone())?;
                        // }
                    }
                    Expr::Wildcard => (),
                    Expr::Slice { input, offset, length } => {
                        input.$visitable_fn(state, f.clone())?;
                        offset.$visitable_fn(state, f.clone())?;
                        length.$visitable_fn(state, f.clone())?;
                    }
                    Expr::Exclude(input, _) => input.$visitable_fn(state, f.clone())?,
                    Expr::KeepName(input) => input.$visitable_fn(state, f.clone())?,
                    Expr::Count => (),
                    Expr::Nth(_) => (),
                    expr => {
                        return Err(Status::invalid_argument(format!(
                            "Logical plan contains unsupported variants: {:?}",
                            expr
                        )))
                    }
                }
                f(self, state)?;
                Ok(())
            }
        }

    )*};
}

visitable_impl_expr! {
    Visitable visit &Self
    VisitableMut visit_mut &mut Self
}
