use polars::prelude::*;

use crate::errors::Result;

pub trait CollectDFCacheExt {
    fn collect_or_get_cached(self) -> Result<Arc<DataFrame>>;
}
impl CollectDFCacheExt for LazyFrame {
    /// This function is useful because, when the LazyFrame is shallow - meaning, it only has
    /// one node, which is a DataFrameScan - it doesn't copy the DataFrame.
    fn collect_or_get_cached(self) -> Result<Arc<DataFrame>> {
        Ok(match self.logical_plan {
            LogicalPlan::DataFrameScan { df, .. } => df,
            _ => Arc::new(self.collect()?),
        })
    }
}

pub trait IntoLazyFrameExt {
    fn into_lazy_frame(self) -> LazyFrame;
}
impl IntoLazyFrameExt for LogicalPlan {
    fn into_lazy_frame(self) -> LazyFrame {
        self.into()
    }
}
impl IntoLazyFrameExt for Arc<DataFrame> {
    /// This function is useful because DataFrame::lazy(self) consumes the dataframe
    /// and we don't want that! we want to avoid a copy of the dataframe everytime we
    /// need a LazyFrame from an Arc<DataFrame>.
    /// This function therefore makes a LazyFrame that aliases with other Arc<DataFrame>
    /// that currently exist.
    fn into_lazy_frame(self) -> LazyFrame {
        let schema = Arc::new(self.schema());
        LogicalPlan::DataFrameScan {
            df: self,
            schema,
            output_schema: None,
            projection: None,
            selection: None,
        }
        .into()
    }
}
