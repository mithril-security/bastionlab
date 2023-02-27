use super::factor::AggregationFactor;
use crate::{
    aggregation::expr_pass::{AggExprCheckPass, GetNRows, LazyGetNRows},
    errors::{BastionLabPolarsError, Result},
    state_tree::{StateNodeChildren, StateTree},
    utils::IntoLazyFrameExt,
    visit::{self, Visitor, VisitorMut},
};
use log::*;
use polars::prelude::*;
use std::{
    cell::RefCell,
    collections::{hash_map::Entry, HashMap},
    rc::Rc,
};

#[derive(Clone, Debug)]
pub struct AggStats {
    agg_factor: HashMap<String, AggregationFactor>,
    // if None, this is not known.
    nrows: Option<usize>,
}
impl AggStats {
    pub fn from_input(df: &DataFrame, identifier: String) -> Self {
        Self {
            agg_factor: [(identifier, AggregationFactor::from_input())].into(),
            nrows: Some(df.height()),
        }
    }

    pub fn agg_factor(&self) -> &HashMap<String, AggregationFactor> {
        &self.agg_factor
    }

    pub fn merge_least(mut self, other: Self) -> Self {
        for (k, v) in other.agg_factor {
            match self.agg_factor.entry(k) {
                Entry::Occupied(mut entry) => {
                    entry.insert(entry.get().least(v));
                }
                Entry::Vacant(entry) => {
                    entry.insert(v);
                }
            }
        }

        AggStats {
            agg_factor: self.agg_factor,
            nrows: if self.nrows == other.nrows {
                self.nrows
            } else if self.nrows == Some(1) {
                other.nrows
            } else if other.nrows == Some(1) {
                self.nrows
            } else {
                None
            },
        }
    }
}

pub struct AggPlanCheckPass {
    registry: StateTree<AggStats>,
}

impl AggPlanCheckPass {
    pub fn new(registry: StateTree<AggStats>) -> Self {
        Self { registry }
    }

    pub fn into_inner(self) -> Result<AggStats> {
        let node = self.registry.try_unwrap()?;
        let stats = node
            .state
            .into_inner()
            .ok_or(BastionLabPolarsError::BadState)?;
        Ok(stats)
    }
}

impl VisitorMut for AggPlanCheckPass {
    fn visit_logical_plan_mut(&mut self, node: &mut LogicalPlan) -> Result<()> {
        self.registry.next();
        visit::visit_logical_plan_mut(self, node)?;

        // Get the `nrows` from the unary input:
        // - if we know its `nrows` count, return that `nrows` count
        // - if we don't know about it, execute the input logical plan get the answer
        fn get_input_nrows(
            registry: StateTree<AggStats>,
            input_lp: Rc<RefCell<LogicalPlan>>,
        ) -> LazyGetNRows {
            Rc::new(move || {
                let StateNodeChildren::Unary(input_state) = &registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };
                let guard = input_state.state.borrow();
                if let Some(nrows) = guard.as_ref().and_then(|e| e.nrows) {
                    return Ok(nrows);
                }

                let mut input_lp = input_lp.as_ref().borrow_mut();

                if let LogicalPlan::DataFrameScan { df, .. } = &*input_lp {
                    trace!("Frame is shallow for {:?}", input_lp);
                    return Ok(df.height());
                }

                trace!(
                    "Computing logical plan {:?} to get the resulting # of rows",
                    input_lp
                );
                // compute the input param!
                let result = input_lp
                    .clone()
                    .into_lazy_frame()
                    .collect()
                    .map_err(|e| BastionLabPolarsError::LogicalPlanErr(e))?;

                let nrows = result.height();
                *input_lp = result.lazy().logical_plan;

                Ok(nrows)
            })
        }

        match node {
            // DataFrameScan is the input of a graph
            // Leaf node
            LogicalPlan::DataFrameScan { .. } => {}

            // Projection/LocalProjection corresponds to SELECT statements
            // Unary node
            LogicalPlan::Projection {
                ref mut input, // input plan
                expr,          // selected expr list
                ..
            }
            | LogicalPlan::LocalProjection {
                ref mut input, // input plan
                expr,          // selected expr list
                ..
            }
            | LogicalPlan::HStack {
                // HStack is `with_column`, which is like Projection but keeps old rows unchanged
                ref mut input, // input plan
                exprs: expr,   // selected expr list
                ..
            } => {
                let StateNodeChildren::Unary(input_state) = &self.registry.children else {
                        return Err(BastionLabPolarsError::BadState)
                    };
                // input_state is the AggStats from the input.

                trace!("Visiting projection: ({input_state:?}) {expr:?}");

                let input_n_rows = input_state.state.borrow();
                let AggStats {
                    agg_factor: agg_factor_input,
                    nrows: nrows_input,
                } = input_n_rows.as_ref().unwrap().clone();

                let (lazy_get_input_nrows, refcell) = if let Some(el) = nrows_input {
                    (GetNRows::Known(el), None)
                } else {
                    let cell = Rc::new(RefCell::new(input.as_ref().clone()));
                    (
                        GetNRows::Lazy(get_input_nrows(self.registry.clone(), Rc::clone(&cell))),
                        Some(cell),
                    )
                };
                let res = expr
                    .iter()
                    .map(|expr| {
                        let mut pass = AggExprCheckPass::new(lazy_get_input_nrows.clone());
                        pass.visit_expr(expr)?;
                        Ok(pass.finish()?)
                    })
                    .collect::<Result<Vec<_>>>()?;

                if let Some(e) = refcell {
                    let e = e.as_ref().borrow();
                    *input = Box::new((&*e).clone());
                }

                trace!("Visit Projection results: {res:?}");

                let agg_factor_local =
                    AggregationFactor::least_of_many(res.iter().map(|e| e.agg_factor()));
                let mut nrows = res
                    .into_iter()
                    .map(|e| e.into_nrows().into_known())
                    .min() // note: None < Some(_)
                    .unwrap_or(None);

                // with_columns keeps previous columns unchanged
                let is_hstack = matches!(node, LogicalPlan::HStack { .. });
                if is_hstack {
                    nrows = nrows.zip(nrows_input).map(|(a, b)| {
                        if a == 1 {
                            b
                        } else if b == 1 {
                            a
                        } else {
                            usize::min(a, b)
                        }
                    });
                }

                let agg_factor = agg_factor_input
                    .iter()
                    .map(|(id, &factor)| {
                        if is_hstack {
                            // with_columns keeps previous columns unchanged
                            (
                                id.clone(),
                                factor.least(factor.after_projection(agg_factor_local)),
                            )
                        } else {
                            (id.clone(), factor.least(agg_factor_local))
                        }
                    })
                    .collect();
                trace!("Updated agg_factor is {agg_factor:?} for projection (local is {agg_factor_local:?}, before is {agg_factor_input:?}).");

                *self.registry.state.borrow_mut() = Some(AggStats { agg_factor, nrows });
            }

            // Aggregate corresponds to GROUBY statements
            // `input.groupby(keys).agg(aggs)`
            // Unary node
            LogicalPlan::Aggregate {
                input,   // input plan (FROM)
                keys,    // selected keys expr list (BY)
                aggs,    // aggregate expr list
                options, // TODO(security): learn how `options.slice` works and other options
                schema,
                apply,
                maintain_order,
            } => {
                // TODO(security): We DONT do any check on `keys`. Is that OK? I'm not 100% sure.

                // resulting agg_factor is the [input.where(expr).count() for expr in keys].min()
                // resulting nrows is the number of unique values in SELECT `keys` FROM `input`

                // we have to run the plan here. there are no other choices i think. :(

                // TODO(security): inspect `aggs`, reject List aggregations and stuff!

                let StateNodeChildren::Unary(input_state) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };
                // input_state is the AggStats from the input.

                trace!("Visiting aggregation: ({input_state:?}) {aggs:?}");

                let input_n_rows = input_state.state.borrow();
                let AggStats {
                    agg_factor: agg_factor_input,
                    nrows: _nrows_input,
                } = input_n_rows.as_ref().unwrap().clone();

                let mut updated_aggs = aggs.clone();
                updated_aggs.push(count().alias("__count")); // add an aggregation: .agg(pl.count())
                                                             // this aggregation will tell us the bucket sizes for each of the returned groups!

                let mut updated_schema = schema.as_ref().clone();
                updated_schema.with_column("__count".into(), IDX_DTYPE);

                let plan = LogicalPlan::Aggregate {
                    input: input.clone(),
                    keys: keys.clone(),
                    aggs: updated_aggs,
                    options: options.clone(),
                    schema: Arc::new(updated_schema),
                    apply: apply.clone(),
                    maintain_order: *maintain_order,
                };
                trace!("Executing Aggregate: {keys:?} {plan:?}");
                let ldf = plan.into_lazy_frame();

                let mut ret = ldf
                    .collect()
                    .map_err(BastionLabPolarsError::LogicalPlanErr)?;

                let count = ret
                    .drop_in_place("__count")
                    .map_err(BastionLabPolarsError::LogicalPlanErr)?;

                let count: usize = count.min().ok_or(BastionLabPolarsError::BadState)?;

                let agg_factor = agg_factor_input
                    .iter()
                    .map(|(id, &factor)| (id.clone(), factor.aggregate_lines(count)))
                    .collect();
                let nrows = ret.height();

                trace!("Count is {count:?} agg_factor is {agg_factor:?}");

                // cache the result for the rest of the graph!
                *node = ret.lazy().logical_plan;

                *self.registry.state.borrow_mut() = Some(AggStats {
                    agg_factor,
                    nrows: Some(nrows),
                });
            }

            // Selection corresponds to WHERE / FILTER statements
            // Distinct corresponds to UNIQUE / UNIQUE BY statements
            // For Slice, you may think that you can know statically the number of elements returned by the operation
            //  ie .slice(0, 10) => 10 elements; but this is false! .slice(0, 10) may actually return anywhere
            //  from 0 to 10 elements, depending on the length of the input dataframe. In this sense, we have to treat
            //  it like a Selection.
            // Unary node
            LogicalPlan::Selection {
                input: _,     // input plan
                predicate: _, // predicate expr
            }
            | LogicalPlan::Slice {
                input: _,  // input plan
                offset: _, // start index
                len: _,    // len
            }
            | LogicalPlan::Distinct {
                input: _, // input plan
                options: _,
            } => {
                let StateNodeChildren::Unary(input_state) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };
                // input_state is the AggStats from the input.

                // TODO(security): We DONT do any check on `predicate`. Is that OK? I'm not 100% sure.

                trace!("Visiting selection: ({input_state:?})");

                let input_n_rows = input_state.state.borrow();
                let AggStats {
                    agg_factor: agg_factor_input,
                    nrows: _nrows_input,
                } = input_n_rows.as_ref().unwrap().clone();

                *self.registry.state.borrow_mut() = Some(AggStats {
                    agg_factor: agg_factor_input,
                    nrows: None, // we have no idea how many rows we have anymore after this point.
                });
            }

            LogicalPlan::Join {
                input_left,
                input_right,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                let StateNodeChildren::Binary(input_state_left, input_state_right) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };

                // TODO(security): We DONT do any check on `left_on` and `right_on`. Is that OK? I'm not 100% sure.
                let (left_scaling, right_scaling, nrows) = match options.how {
                    // Semi Join: Return rows from left that have a match in right.
                    // Cannot return duplicates.
                    // Anti Join: Return rows from left that don't have a match in right.
                    // Cannot return duplicates.
                    JoinType::Semi | JoinType::Anti => (1, 1, None),

                    // Left Join: Return rows from left with their match in right, or null if not found.
                    // Can duplicate right's rows.
                    // TODO make a fast path that doesnt need input_left indices?
                    // Inner Join: Return every pair that matches between left and right.
                    // Can duplicate right's and left's rows.
                    // Outer Join: Return left's rows with their match in right or null, and right rows with their match in left or null; without duplicate pairs.
                    // Can duplicate right's and left's rows.
                    // Cross Join: Carthesian product: will match every row of left with every row of right. Produces exactly `left.nrows * right.nrows` rows.
                    // Can duplicate right's and left's rows.
                    // TODO make a fast path because we don't actually need row counts!
                    // AsOf Join: TODO
                    JoinType::Left
                    | JoinType::Inner
                    | JoinType::Outer
                    | JoinType::Cross
                    | JoinType::AsOf(_) => {
                        let plan_left = input_left
                            .as_ref()
                            .clone()
                            .into_lazy_frame()
                            .with_row_count("__index_left", None);
                        let plan_right = input_right
                            .as_ref()
                            .clone()
                            .into_lazy_frame()
                            .with_row_count("__index_right", None);

                        let ldf = plan_left
                            .join_builder()
                            .left_on(left_on)
                            .right_on(right_on)
                            .with(plan_right)
                            .allow_parallel(options.allow_parallel)
                            .force_parallel(options.force_parallel)
                            .how(options.how.clone())
                            .suffix(options.suffix.clone())
                            .finish();

                        trace!("Executing Join: {:?}", ldf.logical_plan);

                        let ret = ldf
                            .collect()
                            .map_err(BastionLabPolarsError::LogicalPlanErr)?;

                        let get_scaling = |row: &str, other: &str| -> Result<_> {
                            let plan = ret
                                .clone()
                                .lazy()
                                .groupby([col(row)])
                                .agg([col(other).count()])
                                .select([col(other).max()]);
                            trace!("Executing Join (scaling): {:?}", plan.logical_plan);

                            let scaling = plan
                                .collect()
                                .map_err(BastionLabPolarsError::LogicalPlanErr)?
                                .get(0)
                                .and_then(|e| e.get(0).cloned())
                                .ok_or(BastionLabPolarsError::BadState)?
                                .try_extract::<usize>()
                                .map_err(BastionLabPolarsError::LogicalPlanErr)?;
                            Ok(scaling)
                        };

                        let left_scaling = get_scaling("__index_left", "__index_right")?;
                        let right_scaling = get_scaling("__index_right", "__index_left")?;

                        let nrows = ret.height();
                        *node = ret.lazy().logical_plan;

                        (left_scaling, right_scaling, Some(nrows))
                    }
                };

                // merge the dataframe stats!
                let state_left = input_state_left.state.borrow();
                let state_right = input_state_right.state.borrow();
                let map_left = &state_left
                    .as_ref()
                    .ok_or(BastionLabPolarsError::BadState)?
                    .agg_factor;
                let map_right = &state_right
                    .as_ref()
                    .ok_or(BastionLabPolarsError::BadState)?
                    .agg_factor;

                let mut agg_factor_out = HashMap::<String, AggregationFactor>::new();
                for ((k, &v), left) in map_left
                    .iter()
                    .map(|el| (el, true))
                    .chain(map_right.iter().map(|el| (el, false)))
                {
                    let v = if left {
                        v.duplicate_lines(left_scaling)
                    } else {
                        v.duplicate_lines(right_scaling)
                    };
                    match agg_factor_out.entry(k.into()) {
                        Entry::Occupied(mut entry) => {
                            entry.insert(entry.get().least(v));
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(v);
                        }
                    }
                }

                *self.registry.state.borrow_mut() = Some(AggStats {
                    agg_factor: agg_factor_out,
                    nrows,
                });
            }

            // https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.melt.html
            // Unary node
            LogicalPlan::Melt {
                input: _,
                args,
                schema,
            } => {
                // Melt makes nrows_input * value_vars.len()  rows
                // agg_factor for value_vars is unchanged, but
                // id_column values are repeated value_vars.len() times
                // so k := k, duplicate_factor *= value_vars.len().
                let StateNodeChildren::Unary(input_state) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };

                trace!("Visiting melt: ({input_state:?}) ARGS {args:?}");

                let input_n_rows = input_state.state.borrow();
                let AggStats {
                    agg_factor: agg_factor_input,
                    nrows: nrows_input,
                } = input_n_rows.as_ref().unwrap().clone();

                // if `value_vars` is empty; we should treat it as being every column that is not in `id_vars`
                let value_vars_len = if !args.value_vars.is_empty() {
                    args.value_vars.len()
                } else {
                    schema.iter().count() - args.id_vars.len()
                };

                if value_vars_len == 0 {
                    Err(BastionLabPolarsError::AnalysisFailed(format!(
                        "No value vars in MELT operation."
                    )))?
                }

                let agg_factor = agg_factor_input
                    .iter()
                    .map(|(id, &factor)| (id.clone(), factor.duplicate_lines(value_vars_len)))
                    .collect();
                *self.registry.state.borrow_mut() = Some(AggStats {
                    agg_factor,
                    nrows: nrows_input.map(|nrows| nrows * value_vars_len),
                });
            }

            // Unary - do nothing
            LogicalPlan::Sort { .. } | LogicalPlan::Cache { .. } => {
                // nrows and agg_factor is same as input!
                let StateNodeChildren::Unary(input_state) = &self.registry.children else {
                    return Err(BastionLabPolarsError::BadState)
                };
                *self.registry.state.borrow_mut() = input_state.state.borrow().clone();
            }

            op => Err(BastionLabPolarsError::AnalysisFailed(
                format!("Unsupported operation for aggregation analysis: {op:?}").into(),
            ))?,
        }

        self.registry.prev();

        Ok(())
    }

    // Do not recurse on Expr and AggExpr
    fn visit_expr_mut(&mut self, _node: &mut Expr) -> Result<()> {
        Ok(())
    }
    fn visit_agg_expr_mut(&mut self, _node: &mut AggExpr) -> Result<()> {
        Ok(())
    }
}
