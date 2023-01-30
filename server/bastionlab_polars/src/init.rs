use polars::prelude::*;

use crate::{
    composite_plan::ExtendedDataFrame,
    errors::BastionLabPolarsError,
    visit::{self, VisitorMut},
};

pub struct Initializer<'a, S> {
    stack: &'a mut Vec<ExtendedDataFrame<S>>,
    initial_states: Vec<S>,
}

impl<'a, S> Initializer<'a, S> {
    pub fn new(stack: &'a mut Vec<ExtendedDataFrame<S>>) -> Self {
        Initializer {
            stack,
            initial_states: Vec::new(),
        }
    }

    pub fn initial_states(mut self) -> Vec<S> {
        self.initial_states.reverse();
        self.initial_states
    }
}

impl<'a, S> VisitorMut for Initializer<'a, S> {
    fn visit_logical_plan_mut(
        &mut self,
        node: &mut LogicalPlan,
    ) -> Result<(), BastionLabPolarsError> {
        if let LogicalPlan::DataFrameScan { .. } = node {
            let xdf = self.stack.pop().ok_or(BastionLabPolarsError::EmptyStack)?;
            self.initial_states.push(xdf.extension);
            *node = xdf.df.lazy().logical_plan;
        }

        visit::visit_logical_plan_mut(self, node)
    }

    // Do not recurse on Expr
    fn visit_expr_mut(&mut self, _node: &mut Expr) -> Result<(), BastionLabPolarsError> {
        Ok(())
    }

    // Do not recurse on AggExpr
    fn visit_agg_expr_mut(&mut self, _node: &mut AggExpr) -> Result<(), BastionLabPolarsError> {
        Ok(())
    }
}
