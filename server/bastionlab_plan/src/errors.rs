use polars::prelude::PolarsError;
use thiserror::Error;

pub(crate) type Result<T, E = BastionLabPolarsError> = std::result::Result<T, E>;

#[derive(Error, Debug)]
pub enum BastionLabPolarsError {
    #[error("Could not run logical plan: not enough input data frames")]
    EmptyStack,
    #[error("Could not run logical plan: bad state")]
    BadState,
    #[error("Could not get ids of DataFrame `{0}`")]
    UnavailableIds(String),
    #[error("LogicalPlan contains unsupported variant")]
    UnsupportedLogicalPlanVariant,
    #[error("Expr contains unsupported variant")]
    UnsupportedExprVariant,
    #[error("Could not run logical plan: {0}")]
    LogicalPlanErr(#[from] PolarsError),
    #[error("Could not find DataFrame `{0}`")]
    NotFound(String),
    #[error("Could not compute left join scaling")]
    LeftJoinScalingErr,
    #[error("Could not compute right join scaling")]
    RightJoinScalingErr,
    #[error("Could not compute join dependencies")]
    JoinDependenciesErr,
    #[error("Usupported operator `{0}`")]
    UnsupportedOperator(String),
    #[error("Expression analysis failed: {0}")]
    AnalysisFailed(String),
}
