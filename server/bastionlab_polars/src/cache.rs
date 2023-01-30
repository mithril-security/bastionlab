use polars::prelude::*;

use crate::{visit::{self, VisitorMut}, errors::BastionLabPolarsError, state_tree::{StateTree, StateNodeChildren}};

#[derive(Clone)]
pub struct Cache(pub StateTree<LazyFrame>);

impl Cache {
    // fn cache_unary(&mut self, ldf: LazyFrame) -> LazyFrame {}
    pub fn cache_binary(&mut self, left_ldf: LazyFrame, right_ldf: LazyFrame) -> Result<(LazyFrame, LazyFrame), BastionLabPolarsError> {
        let StateNodeChildren::Binary(left_cache, right_cache) = &self.0.children else {
            return Err(BastionLabPolarsError::BadState)
        };

        let left_ldf = left_ldf.collect().map_err(|e| BastionLabPolarsError::LogicalPlanErr(e))?.lazy();
        let right_ldf = right_ldf.collect().map_err(|e| BastionLabPolarsError::LogicalPlanErr(e))?.lazy();

        *left_cache.state.borrow_mut() = Some(left_ldf.clone());
        *right_cache.state.borrow_mut() = Some(right_ldf.clone());

        Ok((left_ldf, right_ldf))
    }
}

impl VisitorMut for Cache {
    fn visit_logical_plan_mut(&mut self, node: &mut LogicalPlan) -> Result<(), BastionLabPolarsError> {
        self.0.next();
        visit::visit_logical_plan_mut(self, node)?;

        if let Some(ldf) = self.0.state.borrow_mut().take() {
            *node = ldf.logical_plan;
        }

        self.0.prev();
        Ok(())
    }
}
