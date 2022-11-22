use polars::prelude::LogicalPlan;
use tonic::Status;

pub trait Visitable<S> {
    fn visit<F: Fn(&Self, &mut S) -> Result<(), Status> + Clone>(&self, state: &mut S, f: F) -> Result<(), Status>;
}

pub trait VisitableMut<S> {
    fn visit_mut<F: Fn(&mut Self, &mut S) -> Result<(), Status> + Clone>(&mut self, state: &mut S, f: F) -> Result<(), Status>;
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
