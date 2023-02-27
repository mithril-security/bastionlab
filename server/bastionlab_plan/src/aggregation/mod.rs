pub mod expr_pass;
pub mod factor;
pub mod plan_pass;

#[cfg(test)]
mod tests {
    use crate::{
        composite_plan::{AnalysisMode, CompositePlan, CompositePlanSegment, State},
        errors::Result,
        visit::{self, VisitorMut},
    };
    use log::*;
    use polars::{
        io::mmap::MmapBytesReader,
        lazy::dsl::{when, StrpTimeOptions},
        prelude::*,
    };
    use std::{fs::File, ops::Mul};

    fn init() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Trace)
            .try_init();
    }

    fn get_titanic() -> DataFrame {
        let file = Box::new(File::open("./titanic.csv").unwrap()) as Box<dyn MmapBytesReader>;
        let df = CsvReader::new(file).finish().unwrap();
        df
    }

    fn get_covid() -> DataFrame {
        let file = Box::new(File::open("./covid_19_data.csv").unwrap()) as Box<dyn MmapBytesReader>;
        let df = CsvReader::new(file).finish().unwrap();
        df
    }

    fn run_analysis_auto(mut plan: LazyFrame, expected: impl AsRef<[(usize, usize)]>) {
        struct FindInputFrames(Vec<(String, Arc<DataFrame>)>, usize);
        impl VisitorMut for FindInputFrames {
            fn visit_logical_plan_mut(&mut self, node: &mut LogicalPlan) -> Result<()> {
                visit::visit_logical_plan_mut(self, node)?;

                if let LogicalPlan::DataFrameScan { ref mut df, .. } = node {
                    self.0.push((format!("input_{}", self.1), df.clone()));
                    self.1 += 1;
                    *df = Default::default();
                }

                Ok(())
            }
            fn visit_agg_expr_mut(&mut self, _node: &mut AggExpr) -> Result<()> {
                Ok(())
            }
            fn visit_expr_mut(&mut self, _node: &mut Expr) -> Result<()> {
                Ok(())
            }
        }

        let mut visitor = FindInputFrames(Default::default(), 0);
        visitor
            .visit_logical_plan_mut(&mut plan.logical_plan)
            .unwrap();
        let map = visitor.0;

        let mut vec = vec![];
        for (k, _) in map.iter().rev() {
            vec.push(CompositePlanSegment::EntryPointPlanSegment {
                identifier: k.into(),
            })
        }
        vec.push(CompositePlanSegment::PolarsPlanSegment {
            plan: plan.logical_plan,
        });

        let plan = CompositePlan { segments: vec };

        let (_df, ext) = plan
            .run(
                State(map.into_iter().collect()),
                AnalysisMode::AggregationCheck,
            )
            .unwrap();

        let stats = ext.agg_stats.unwrap();
        let result = stats.agg_factor();

        let expected = expected.as_ref();
        let mut vec: Vec<_> = result.iter().collect();
        vec.sort_by_key(|e| e.0);
        for ((key, factor), expected) in vec.iter().zip(expected) {
            let val = (factor.k(), factor.duplicate_factor());
            trace!("Analysis result: {key} {factor:?} (expected {expected:?})");
            assert_eq!(val, *expected);
        }
    }

    #[test] // PASS
    fn test_run_simple_plan() {
        init();
        run_analysis_auto(
            get_titanic()
                .lazy()
                .select(&[col("Pclass"), col("Survived")])
                .select(&[col("Pclass").mean(), col("Survived").mean()]),
            [(891, 1)],
        );
        // run_analysis_auto(get_titanic().lazy().select(&[all().head(Some(5))]), [(1, 1)]);
        // run_analysis_auto(
        //     get_titanic()
        //         .lazy()
        //         .select(&[col("Pclass").head(Some(5)), col("Survived")]),
        //     [(1, 1)],
        // );
    }

    #[test] // PASS
    fn test_groupby() {
        init();
        run_analysis_auto(
            get_titanic()
                .lazy()
                .select(&[col("Pclass"), col("Survived")])
                .groupby(&[col("Pclass")])
                .agg(&[col("Survived").mean()])
                .sort("Survived", Default::default()),
            [(184, 1)],
        );
    }

    #[test] // PASS
    fn test_filter() {
        init();
        run_analysis_auto(get_titanic().lazy().filter(col("Pclass").eq(1)), [(1, 1)]);
        run_analysis_auto(
            get_titanic()
                .lazy()
                .select([col("Survived"), col("Age").clip(0.into(), 50.into())])
                .filter(col("Survived").eq(1)),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_filter_agg() {
        init();
        // followed by aggregation (needs nrows! => have to compute part of the graph)
        run_analysis_auto(
            get_titanic().lazy().filter(col("Pclass").eq(1)).sum(),
            [(216, 1)],
        );
    }

    #[test] // PASS
    fn test_null_total() {
        init();
        run_analysis_auto(
            get_titanic().lazy().select([col("Age").is_null().sum()]),
            [(891, 1)],
        );
    }

    #[test] // PASS
    fn test_fill_null() {
        init();
        run_analysis_auto(
            get_titanic()
                .lazy()
                .fill_null(lit("100"))
                .select(&[col("Age").is_null().sum()]),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_hstack_cast() {
        init();
        run_analysis_auto(
            get_titanic()
                .lazy()
                .with_column(col("Age").cast(DataType::Int64))
                .select(&[col("Age").mean()]),
            [(891, 1)],
        );
    }

    #[test] // PASS
    fn test_drop_nulls() {
        init();
        run_analysis_auto(get_titanic().lazy().drop_nulls(None), [(1, 1)]);
    }

    #[test] // PASS
    fn test_unique() {
        init();
        run_analysis_auto(
            get_titanic()
                .lazy()
                .unique(None, UniqueKeepStrategy::First)
                .select(&[col("Sex")]),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_ternary() {
        init();
        run_analysis_auto(
            get_titanic().lazy().select(&[when(col("Sex").eq(lit("M")))
                .then(lit("male"))
                .when(col("Sex").eq(lit("m")))
                .then(lit("male"))
                .when(col("Sex").eq(lit("m")))
                .then(lit("male"))
                .when(col("Sex").eq(lit("F")))
                .then(lit("female"))
                .when(col("Sex").eq(lit("f")))
                .then(lit("female"))
                .otherwise(col("Sex"))]),
            [(1, 1)],
        );
    }

    // #[test]
    // fn test_vstack() {
    //     init();
    //     let df1 = df!(
    //         "Element"=> ["Copper", "Silver", "Silver", "Gold"],
    //         "Melting Point (K)"=> [1357.77, 1234.93, 1234.93, 1337.33],
    //     )
    //     .unwrap();
    //     let df2 = df!(
    //         "Element"=> ["Platinum", "Palladium"],
    //         "Melting Point (K)"=> [2041.4, 1828.05],
    //     )
    //     .unwrap();
    //     let df3 = df!(
    //         "Element"=> ["Titanium"],
    //         "Melting Point (K)"=> [1945.0],
    //     )
    //     .unwrap();
    //     run_analysis_auto(df1.lazy().vstack(rdf2), 891);
    // }

    #[test] // PASS
    fn test_join() {
        init();
        let df1 = df!(
            "Element"=> ["Copper", "Silver", "Silver", "Gold"],
            "Melting Point (K)"=> [1357.77, 1234.93, 1234.93, 1337.33],
        )
        .unwrap();
        let df2 = df!(
            "Element" => ["Magnesium", "Silver", "Gold", "Platinum"],
            "Symbol" => ["Mg", "Ag", "Au", "Pt"],
            "Number" => [12, 47, 79, 78],
        )
        .unwrap();

        run_analysis_auto(
            df1.clone()
                .lazy()
                .join_builder()
                .left_on(&[col("Element")])
                .right_on(&[col("Element")])
                .with(df2.clone().lazy())
                .how(JoinType::Inner)
                .finish(),
            // scaling is two (at most 2 lines are duplicates), so denominator is two
            [(1, 2), (1, 1)],
        );
        run_analysis_auto(
            df1.clone()
                .lazy()
                .join_builder()
                .left_on(&[col("Element")])
                .right_on(&[col("Element")])
                .with(df2.clone().lazy())
                .how(JoinType::Anti)
                .finish(),
            // anti cannot change scaling
            [(1, 1), (1, 1)],
        );
    }

    #[test] // PASS
    fn test_join_asof() {
        init();
        let df1 = df!(
            "distance" => [7, 16, 24, 49],
            "name" => ["Laura", "Charles", "Kwabena", "Shannon"],
        )
        .unwrap();
        let df2 = df!(
            "distance" => [1, 10, 25, 50],
            "level unlocked" => ["Amateur", "Intermediate", "Excellent", "Pro"],
        )
        .unwrap();

        run_analysis_auto(
            df1.clone()
                .lazy()
                .join_builder()
                .left_on(&[col("distance")])
                .right_on(&[col("distance")])
                .with(df2.clone().lazy())
                .how(JoinType::AsOf(AsOfOptions {
                    strategy: AsofStrategy::Forward,
                    left_by: None,
                    right_by: None,
                    tolerance: None,
                    tolerance_str: None,
                }))
                .finish(),
            [(1, 2)],
        );
    }

    #[test] // PASS
    fn test_scaling_zscore() {
        init();
        let df1 = df!(
            "Col A" => [180000, 360000, 230000, 60000],
            "Col B" => [110, 905, 230, 450],
            "Col C" => [18.9, 23.4, 14.0, 13.5],
            "Col D" => [1400, 1800, 1300, 1500]
        )
        .unwrap();

        // zscore
        run_analysis_auto(
            df1.clone().lazy().with_columns(
                df1.get_column_names()
                    .into_iter()
                    .map(|x| (col(x) - col(x).mean()) / col(x).std(0).alias(x))
                    .collect::<Vec<_>>(),
            ),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_scaling_melt() {
        init();
        let df1 = df!(
            "Row" => [0, 1, 2, 3],
            "Col A" => [180000, 360000, 230000, 60000],
            "Col B" => [110, 905, 230, 450],
            "Col C" => [18.9, 23.4, 14.0, 13.5],
            "Col D" => [1400, 1800, 1300, 1500]
        )
        .unwrap();
        // melt
        run_analysis_auto(
            df1.clone()
                .lazy()
                // .with_row_count("Row", None)
                .melt(MeltArgs {
                    id_vars: vec!["Row".into()],
                    value_vars: vec![
                        "Col A".into(),
                        "Col B".into(),
                        "Col C".into(),
                        "Col D".into(),
                    ],
                    variable_name: Some("Column".into()),
                    value_name: Some("Value".into()),
                }),
            [(1, 4)], // 1/4 because we melt 4 columns (id row gets dup 4 times)
        );
    }

    #[test] // PASS
    fn test_scaling_minmax() {
        init();
        let df1 = df!(
            "Col A" => [180000, 360000, 230000, 60000],
            "Col B" => [110, 905, 230, 450],
            "Col C" => [18.9, 23.4, 14.0, 13.5],
            "Col D" => [1400, 1800, 1300, 1500]
        )
        .unwrap();
        // min/max scaling
        run_analysis_auto(
            df1.clone().lazy().with_columns(
                df1.get_column_names()
                    .into_iter()
                    .map(|x| (col(x) - col(x).min()) / (col(x).max() - col(x).min()).alias(x))
                    .collect::<Vec<_>>(),
            ),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_scaling_maxabs() {
        init();
        let df1 = df!(
            "Col A" => [180000, 360000, 230000, 60000],
            "Col B" => [110, 905, 230, 450],
            "Col C" => [18.9, 23.4, 14.0, 13.5],
            "Col D" => [1400, 1800, 1300, 1500]
        )
        .unwrap();
        let abs = |expr: Expr| {
            when(expr.clone().gt_eq(lit(0)))
                .then(expr.clone())
                .otherwise(expr.mul(lit(-1)))
        };
        // max abs scaling
        run_analysis_auto(
            df1.clone().lazy().with_columns(
                df1.get_column_names()
                    .into_iter()
                    .map(|x| abs(col(x)) / (abs(col(x)).max()).alias(x))
                    .collect::<Vec<_>>(),
            ),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_scaling_mean() {
        init();
        let df1 = df!(
            "Col A" => [180000, 360000, 230000, 60000],
            "Col B" => [110, 905, 230, 450],
            "Col C" => [18.9, 23.4, 14.0, 13.5],
            "Col D" => [1400, 1800, 1300, 1500]
        )
        .unwrap();
        // mean scaling
        run_analysis_auto(
            df1.clone().lazy().with_columns(
                df1.get_column_names()
                    .into_iter()
                    .map(|x| (col(x) - col(x).mean()) / (col(x).max() - col(x).min()).alias(x))
                    .collect::<Vec<_>>(),
            ),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_scaling_median() {
        init();
        let df1 = df!(
            "Col A" => [180000, 360000, 230000, 60000],
            "Col B" => [110, 905, 230, 450],
            "Col C" => [18.9, 23.4, 14.0, 13.5],
            "Col D" => [1400, 1800, 1300, 1500]
        )
        .unwrap();
        // median quantile scaling
        run_analysis_auto(
            df1.clone().lazy().with_columns(
                df1.get_column_names()
                    .into_iter()
                    .map(|x| {
                        (col(x) - col(x).median())
                            / (col(x).quantile(0.75, QuantileInterpolOptions::Lower)
                                - col(x).quantile(0.25, QuantileInterpolOptions::Lower))
                            .alias(x)
                    })
                    .collect::<Vec<_>>(),
            ),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_percent_missing() {
        init();
        let df = get_covid();
        // percent_missing
        run_analysis_auto(
            df.clone().lazy().select(
                df.get_column_names()
                    .into_iter()
                    .map(|x| col(x).is_null().sum() * lit(100) / col(x).count())
                    .collect::<Vec<_>>(),
            ),
            [(236017, 1)],
        );
    }

    #[test] // PASS
    fn test_fill_null_province() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy()
                .with_column(col("Province/State"))
                .fill_null(lit("Unknown"))
                .select(&[col("Province/State").is_null().sum()]),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_cleaning_strptime() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy()
                .with_columns(
                    ["Confirmed", "Deaths", "Recovered"].map(|x| col(x).cast(DataType::Int64)),
                )
                .with_column(col("ObservationDate").str().strptime(StrpTimeOptions {
                    date_dtype: DataType::Date,
                    fmt: Some("%m/%d/%Y".into()),
                    strict: false,
                    exact: false,
                })),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_cleaning_country() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy()
                .with_column(
                    when(col("Country/Region").eq(lit(" Azerbaijan")))
                        .then(lit("Azerbaijan"))
                        .when(col("Country/Region").eq(lit("('St. Martin',)")))
                        .then(lit("St. Martin"))
                        .when(col("Country/Region").str().starts_with("Congo"))
                        .then(lit("Republic of the Congo"))
                        .otherwise(col("Country/Region"))
                        .alias("Country/Region"),
                )
                .select(&[col("Country/Region")])
                .unique(None, UniqueKeepStrategy::First)
                .select(&[col("Country/Region").count()]),
            [(223, 1)],
        );
    }

    #[test] // PASS
    fn test_cleaning_add_column() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy().with_column(
                (col("Confirmed") - col("Deaths") - col("Recovered")).alias("Active_cases"),
            ),
            [(1, 1)],
        );
    }

    #[test] // PASS
    fn test_cleaning_data_analysis() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy()
                .filter(col("ObservationDate").eq(col("ObservationDate").max()))
                .with_column(
                    (col("Confirmed") - col("Deaths") - col("Recovered")).alias("Active_cases"),
                )
                .select(["Deaths", "Confirmed", "Recovered", "Active_cases"].map(|x| col(x).sum())),
            [(761, 1)],
        );
    }

    #[test] // PASS
    fn test_cleaning_g7_cases() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy()
                .filter(
                    (col("Country/Region").eq(lit("Canada")))
                        .or(col("Country/Region").eq(lit("France")))
                        .or(col("Country/Region").eq(lit("Germany")))
                        .or(col("Country/Region").eq(lit("Italy")))
                        .or(col("Country/Region").eq(lit("UK")))
                        .or(col("Country/Region").eq(lit("US")))
                        .or(col("Country/Region").eq(lit("Japan"))),
                )
                .with_columns(["Deaths", "Confirmed", "Recovered"].map(|x| col(x).sum()))
                .filter(col("ObservationDate").eq(col("ObservationDate").max()))
                .select(["Deaths", "Confirmed", "Recovered", "Country/Region"].map(col))
                .groupby(&[col("Country/Region")])
                .agg(&[col("Confirmed").sum()])
                .sort(
                    "Confirmed",
                    SortOptions {
                        descending: false,
                        nulls_last: false,
                    },
                ),
            [(11, 1)], // this is because France has 11 lines (groupby)
        );
    }

    #[test] // PASS
    fn test_cleaning_uk_evolution() {
        init();
        let df = get_covid();
        run_analysis_auto(
            df.lazy()
                .with_column(
                    col("ObservationDate")
                        .str()
                        .strptime(StrpTimeOptions {
                            date_dtype: DataType::Date,
                            fmt: Some("%m/%d/%Y".into()),
                            strict: false,
                            exact: false,
                        })
                        .alias("Date2"),
                )
                .filter(
                    col("Country/Region")
                        .eq(lit("UK"))
                        .and(col("Date2").dt().year().eq(lit(2020)))
                        .and(col("Date2").dt().month().gt(lit(3))),
                )
                .groupby(&[col("Date2").dt().month()])
                .agg(&[col("Confirmed").sum()])
                .sort(
                    "Date2",
                    SortOptions {
                        descending: false,
                        nulls_last: false,
                    },
                ),
            [(354, 1)],
        )
    }
}
