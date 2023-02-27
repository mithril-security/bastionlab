use super::factor::AggregationFactor;
use crate::{
    errors::{BastionLabPolarsError, Result},
    visit::{self, Visitor},
};
use log::*;
use polars::lazy::dsl::{AggExpr, Expr, FunctionExpr};
use std::{fmt::Debug, rc::Rc};

pub type LazyGetNRows = Rc<dyn Fn() -> Result<usize> + 'static>;

#[derive(Clone)]
pub enum GetNRows {
    /// The number of rows is known statically
    Known(usize),
    /// Lazily get the number of rows
    Lazy(LazyGetNRows),
}

impl Debug for GetNRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GetNRows::Known(n) => write!(f, "{}", n),
            GetNRows::Lazy(_) => write!(f, "<lazy>"),
        }
    }
}

impl GetNRows {
    pub fn into_known(self) -> Option<usize> {
        match self {
            Self::Known(e) => Some(e),
            Self::Lazy(_) => None,
        }
    }

    pub fn map(self, f: impl Fn(usize) -> Result<usize> + 'static) -> Result<Self> {
        Ok(match self {
            Self::Known(e) => Self::Known(f(e)?),
            Self::Lazy(e) => Self::Lazy(Rc::new(move || f(e()?))),
        })
    }

    pub fn map_binary(
        self,
        second: Self,
        f: impl Fn(usize, usize) -> Result<usize> + 'static,
    ) -> Result<Self> {
        match self {
            GetNRows::Known(a) => second.map(move |b| f(a, b)),
            GetNRows::Lazy(a) => second.map(move |b| f(a()?, b)),
        }
    }

    pub fn map_ternary(
        self,
        second: Self,
        third: Self,
        f: impl Fn(usize, usize, usize) -> Result<usize> + 'static,
    ) -> Result<Self> {
        match self {
            GetNRows::Known(a) => second.map_binary(third, move |b, c| f(a, b, c)),
            GetNRows::Lazy(a) => second.map_binary(third, move |b, c| f(a()?, b, c)),
        }
    }

    pub fn evaluate(self) -> Result<usize> {
        match self {
            GetNRows::Known(n) => Ok(n),
            GetNRows::Lazy(f) => f(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct ExprAgg {
    agg_factor: AggregationFactor,
    nrows: GetNRows,
}

impl ExprAgg {
    /// result of a binary expr between self and other
    fn merge(agg_factors: &[AggregationFactor], nrows: GetNRows) -> Self {
        let res = Self {
            agg_factor: AggregationFactor::least_of_many(agg_factors),
            nrows,
        };

        trace!("Merge result {res:?}");
        res
    }

    fn from_input(agg_factor: AggregationFactor, nrows: GetNRows) -> Self {
        ExprAgg { agg_factor, nrows }
    }

    fn litteral() -> Self {
        ExprAgg {
            agg_factor: AggregationFactor::from_constant(),
            nrows: GetNRows::Known(1),
        }
    }

    fn aggregation(self) -> Result<Self> {
        let nrows = self.nrows.evaluate()?;

        let res = Self {
            agg_factor: self.agg_factor.aggregate_lines(nrows),
            nrows: GetNRows::Known(1),
        };
        trace!("Aggregation result {res:?}");
        Ok(res)
    }

    pub fn agg_factor(&self) -> AggregationFactor {
        self.agg_factor
    }

    pub fn into_nrows(self) -> GetNRows {
        self.nrows
    }

    pub fn nrows(&self) -> &GetNRows {
        &self.nrows
    }
}

#[derive(Clone)]
pub struct AggExprCheckPass {
    stack: Vec<ExprAgg>,
    input_n_rows: GetNRows,
}

impl AggExprCheckPass {
    pub fn new(input_n_rows: GetNRows) -> Self {
        Self {
            input_n_rows,
            stack: Vec::new(),
        }
    }

    pub fn finish(mut self) -> Result<ExprAgg> {
        self.stack.pop().ok_or(BastionLabPolarsError::BadState)
    }
}

impl Visitor for AggExprCheckPass {
    fn visit_expr(&mut self, node: &Expr) -> Result<()> {
        visit::visit_expr(self, node)?;

        let mut stack_pop = || self.stack.pop().ok_or(BastionLabPolarsError::BadState);

        match node {
            // Variants choose specific columns from the dataframe input
            Expr::Column(_)
            | Expr::Columns(_)
            | Expr::DtypeColumn(_)
            | Expr::Wildcard
            | Expr::Nth(_) => self.stack.push(ExprAgg::from_input(
                AggregationFactor::from_input(),
                self.input_n_rows.clone(),
            )),
            // Variants independant from data
            Expr::Literal(_) | Expr::Count => self.stack.push(ExprAgg::litteral()),
            // True Aggregation, Unary
            Expr::Agg(
                AggExpr::Mean(_) | AggExpr::Sum(_) | AggExpr::Std(_, _) | AggExpr::Var(_, _),
            )
            | Expr::Function {
                function: FunctionExpr::NullCount,
                ..
            } => {
                let inp = stack_pop()?;
                self.stack.push(inp.aggregation()?);
            }
            // More aggregation
            // TODO: Shoul `AggExpr::Count` be a literal?
            Expr::Agg(
                AggExpr::Quantile { .. }
                | AggExpr::Min { .. }
                | AggExpr::Max { .. }
                | AggExpr::Median(_)
                | AggExpr::Count(_),
            ) => {
                let inp = stack_pop()?;
                self.stack.push(inp.aggregation()?);
            }
            // Binary expression
            Expr::BinaryExpr { .. }
            | Expr::Function {
                function: FunctionExpr::Pow | FunctionExpr::FillNull { .. },
                ..
            } => {
                let rhs = stack_pop()?;
                let lhs = stack_pop()?;

                let nrows =
                    GetNRows::map_binary(lhs.nrows, rhs.nrows, |mut lhs_count, mut rhs_count| {
                        if rhs_count == 1 {
                            rhs_count = lhs_count;
                        }
                        if lhs_count == 1 {
                            lhs_count = rhs_count;
                        }
                        Ok(usize::min(rhs_count, lhs_count))
                    })?;

                self.stack
                    .push(ExprAgg::merge(&[lhs.agg_factor, rhs.agg_factor], nrows));
            }
            // If predicate Then expr1 Else expr2 expression
            Expr::Ternary { .. } => {
                let expr2 = stack_pop()?;
                let expr1 = stack_pop()?;
                let predicate = stack_pop()?;

                let nrows =
                    GetNRows::map_binary(expr1.nrows, expr2.nrows, |a, b| Ok(usize::min(a, b)))?;

                self.stack.push(ExprAgg::merge(
                    &[predicate.agg_factor, expr1.agg_factor, expr2.agg_factor],
                    nrows,
                ));
            }
            // RT Dependent
            // Expr::Filter { .. } => {
            //     let _by = stack_pop()?;
            //     let input = stack_pop()?;

            //     unimplemented!()
            // }
            // Expr::Slice { .. } => {
            //     let _length = stack_pop()?;
            //     let _offset = stack_pop()?;
            //     let input = stack_pop()?;

            //     unimplemented!()
            // }
            /*
            | FunctionExpr::StringExpr(
                StringFunction::Contains { .. }
                | StringFunction::EndsWith(_)
                | StringFunction::StartsWith(_)
                | StringFunction::Extract { .. }
                | StringFunction::ExtractAll { .. }
                | StringFunction::StartsWith(_)
                | StringFunction::ExtractAll(_)
                | StringFunction::CountMatch(_)
                | StringFunction::Zfill(_)
                | StringFunction::LJust { .. }
                | StringFunction::RJust { .. }
                | StringFunction::Strptime(_)
                | StringFunction::ConcatVertical(_)
                | StringFunction::ConcatHorizontal(_)
                | StringFunction::ConcatVertical(_)
                | StringFunction::Uppercase { .. }
                | StringFunction::Lowercase { .. }
                | StringFunction::Strip(_)
                | StringFunction::LStrip(_)
                | StringFunction::RStrip(_),
            )
            | FunctionExpr::TemporalExpr(
                TemporalFunction::Year
                | TemporalFunction::IsoYear
                | TemporalFunction::Month
                | TemporalFunction::Quarter
                | TemporalFunction::Week
                | TemporalFunction::WeekDay
                | TemporalFunction::Day
                | TemporalFunction::OrdinalDay
                | TemporalFunction::Hour
                | TemporalFunction::Minute
                | TemporalFunction::Second
                | TemporalFunction::Millisecond
                | TemporalFunction::Microsecond
                | TemporalFunction::Nanosecond
                | TemporalFunction::TimeStamp(_)
                | TemporalFunction::Truncate(_)
                | TemporalFunction::Round(_, _),
            ), */
            // Unary
            Expr::Cast { .. }
            | Expr::Alias { .. }
            | Expr::KeepName { .. }
            | Expr::Exclude { .. }
            | Expr::Sort { .. }
            | Expr::Function {
                function:
                    FunctionExpr::IsNull
                    | FunctionExpr::IsNotNull
                    | FunctionExpr::Clip { .. }
                    | FunctionExpr::Hash(_, _, _, _)
                    | FunctionExpr::IsUnique
                    | FunctionExpr::IsDuplicated
                    | FunctionExpr::ShrinkType
                    | FunctionExpr::TemporalExpr(_)
                    | FunctionExpr::StringExpr(_),
                ..
            } => {
                // do nothing; properties are the same as input
            }
            op => Err(BastionLabPolarsError::AnalysisFailed(
                format!("Unsupported expr for aggregation analysis: {op:?}").into(),
            ))?,
        }

        Ok(())
    }
}
