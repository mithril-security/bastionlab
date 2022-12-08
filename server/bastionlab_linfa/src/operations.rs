use linfa::traits::Predict;
use ndarray::{Array2, Dim, Ix2, Shape, ShapeBuilder, StrideShape};

use polars::{
    df,
    error::ErrString,
    prelude::{DataFrame, NamedFrom, PolarsError, PolarsResult},
    series::Series,
};
use tonic::Status;

use crate::{
    algorithms::{decision_trees, elastic_net, gaussian_naive_bayes, kmeans, linear_regression},
    to_status_error,
    trainer::{get_datasets, to_polars_error, Models, PredictionTypes, SupportedModels},
};

use super::algorithms::logistic_regression;

#[macro_export]
macro_rules! to_type {
    {<$t_ty:ty>($item:ident)} => {
        // Use specified type.
        $item.iter().map(|v| *v as $t_ty).collect::<Vec<$t_ty>>()
    };
}

#[macro_export]
macro_rules! to_ndarray {
    ($sh:ident, $target:ident) => {
        to_polars_error(Array2::from_shape_vec($sh, $target))?
            .as_standard_layout()
            .to_owned()
    };
}

fn df_to_vec(df: DataFrame, sh: (usize, usize)) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(sh.0 * sh.1);
    let df = df.get_columns();
    df.iter().for_each(|s| {
        let s = s.cast(&polars::prelude::DataType::Float64).unwrap();
        let s = s.f64().unwrap();
        let s = s.cont_slice().unwrap();
        out.append(&mut s.to_vec());
    });
    out
}

//TODO: Fix Trait implementation for `vec_to_df`

fn vec_f64_to_df(data: Vec<f64>, name: &str) -> PolarsResult<DataFrame> {
    let s = Series::new(name, data);
    let df = to_polars_error(DataFrame::new(vec![s]))?;
    Ok(df)
}

fn vec_u64_to_df(data: Vec<usize>, name: &str) -> PolarsResult<DataFrame> {
    let data = to_type! {<u64>(data)};
    let s = Series::new(name, data);
    let df = to_polars_error(DataFrame::new(vec![s]))?;
    Ok(df)
}

/// Internal structure to hold the [`DataFrame`]s converted to raw [`Vec<f64>`].
/// This also holds information about the features of the input dataset.
///
/// The shapes are stored alongside inorder to effectively use [`Array2::from_shape_vec`].
struct Trainer {
    records: Vec<f64>,
    target: Vec<f64>,
    cols: Vec<String>,
    records_shape: StrideShape<Ix2>,
    target_shape: Shape<Ix2>,
}

/// Transform [`DataFrame`] into [`Trainer`].
///
/// It does so by using the [`df_to_vec`] function which converts [`DataFrame`] into [`Vec<Series>`].
/// Example:
///```
/// shape: (4, 4)
///      ┌──────┬──────┬──────┬──────┐
///      │ col1 ┆ col2 ┆ col3 ┆ col4 │
///      │ ---  ┆ ---  ┆ ---  ┆ ---  │
///      │ i64  ┆ i64  ┆ i64  ┆ i64  │
///      ╞══════╪══════╪══════╪══════╡
///      │ 1    ┆ 3    ┆ 6    ┆ 9    │
///      ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
///      │ 2    ┆ 4    ┆ 8    ┆ 16   │
///      ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
///      │ 3    ┆ 9    ┆ 18   ┆ 35   │
///      ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
///      │ 4    ┆ -1   ┆ 0    ┆ 1    │
///      └──────┴──────┴──────┴──────┘
///```
/// The [`DataFrame`] above is first transformed into [Vec<Series>].
///
/// ```[shape: (4,) Series: 'col1' [i64]
/// [
/// 	1
/// 	2
/// 	3
/// 	4
/// ], shape: (4,)
/// Series: 'col2' [i64]
/// [
/// 	3
/// 	4
/// 	9
/// 	-1
/// ], shape: (4,)
/// Series: 'col3' [i64]
/// [
/// 	6
/// 	8
/// 	18
/// 	0
/// ], shape: (4,)
/// Series: 'col4' [i64]
/// [
/// 	9
/// 	16
/// 	35
/// 	1
/// ]]
/// ```
/// And this [Vec<Series>] is then converted to [Vec<f64>] using an intermediary [Vec<String>].
///
/// **NB: We use, [f64] in order to capture every type (i{8,16,32}, u{8,16,32}, f{32,64}, usize). But, separate
///  conversion path is used when necessary.**
fn transform_dfs(records: DataFrame, target: DataFrame) -> Trainer {
    let cols = records
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let (records_shape, target_shape) = (records.shape(), target.shape());
    let records = df_to_vec(records.clone(), records.shape());
    let target = df_to_vec(target.clone(), target.shape());

    Trainer {
        records,
        target,
        cols,
        records_shape: Shape::from(Dim([records_shape.0, records_shape.1]))
            .strides(Dim([1, records_shape.0])),
        target_shape: Shape::from(Dim([target_shape.0, target_shape.1])),
    }
}

/// This method sends both the training and target datasets to the specified model in [`Models`].
/// And `ratio` is passed along to [`linfa_datasets::DatasetBase`]
pub fn send_to_trainer(
    records: DataFrame,
    target: DataFrame,
    ratio: f32,
    model_type: Models,
) -> PolarsResult<(SupportedModels, (DataFrame, DataFrame))> {
    // We are assuming [`f64`] for all computation since it can represent all other types.
    match model_type {
        Models::GaussianNaiveBayes { var_smoothing } => {
            let var_smoothing: f64 = var_smoothing.into();
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<usize>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = target
                .clone()
                .into_shape([target.clone().len()])
                .map_err(|e| PolarsError::InvalidOperation(ErrString::Owned(e.to_string())))?;

            let records = to_ndarray!(records_shape, records);

            let (train, test) = get_datasets(records, target, ratio, cols)?;
            let (model, _) = to_polars_error(gaussian_naive_bayes(
                train,
                test.clone(),
                var_smoothing.into(),
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_u64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::GaussianNaiveBayes(model), (rec, tar)))
        }
        Models::ElasticNet {
            penalty,
            l1_ratio,
            with_intercept,
            max_iterations,
            tolerance,
        } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<f64>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (train, test) = get_datasets(records, target, ratio, cols)?;
            let model = to_polars_error(elastic_net(
                train,
                test.clone(),
                penalty.into(),
                l1_ratio.into(),
                with_intercept,
                max_iterations,
                tolerance.into(),
            ))?;
            let rec = vec_f64_to_df(test.records.clone().into_raw_vec(), "test_recs")?;
            let tar = vec_f64_to_df(test.targets.clone().into_raw_vec(), "test_target")?;
            Ok((SupportedModels::ElasticNet(model), (rec, tar)))
        }
        Models::KMeans {
            n_runs,
            n_clusters,
            tolerance,
            max_n_iterations,
            init_method,
        } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<f64>(target)};

            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (train, test) = get_datasets(records, target, ratio, cols)?;
            let model = to_polars_error(kmeans(
                train,
                test.clone(),
                n_runs.into(),
                n_clusters.into(),
                tolerance,
                max_n_iterations,
                init_method,
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_f64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::KMeans(model), (rec, tar)))
        }
        Models::LinearRegression { fit_intercept } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<f64>(target)};
            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (dataset, test) = get_datasets(records, target, ratio, cols)?;
            let model = to_polars_error(linear_regression(dataset, fit_intercept))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_f64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::LinearRegression(model), (rec, tar)))
        }

        Models::LogisticRegression {
            alpha,
            gradient_tolerance,
            fit_intercept,
            max_iterations,
            initial_params,
        } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<usize>(target)};
            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (dataset, test) = get_datasets(records, target, ratio, cols)?;

            let model = to_polars_error(logistic_regression(
                dataset,
                alpha,
                gradient_tolerance,
                fit_intercept,
                max_iterations,
                initial_params,
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_u64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::LogisticRegression(model), (rec, tar)))
        }

        Models::DecisionTree {
            split_quality,
            max_depth,
            min_weight_split,
            min_weight_leaf,
            min_impurity_decrease,
        } => {
            let Trainer {
                records,
                target,
                cols,
                records_shape,
                target_shape,
            } = transform_dfs(records, target);

            let target = to_type! {<usize>(target)};
            let target = to_ndarray!(target_shape, target);
            let target = to_polars_error(target.clone().into_shape([target.clone().len()]))?;

            let records = to_ndarray!(records_shape, records);
            let (train, test) = get_datasets(records, target, ratio, cols)?;

            // TODO: Save Test set with the model id.

            let model = to_polars_error(decision_trees(
                train,
                split_quality,
                max_depth,
                min_weight_split,
                min_weight_leaf,
                min_impurity_decrease,
            ))?;

            let rec = vec_f64_to_df(test.records.into_raw_vec(), "test_recs")?;
            let tar = vec_u64_to_df(test.targets.into_raw_vec(), "test_target")?;
            Ok((SupportedModels::DecisionTree(model), (rec, tar)))
        }
    }
}

/// This method is used to run a prediction on an already fitted model, based on the model selection type.
/// We use two different types for prediction
/// [f64] and [usize] --> [PredictionTypes::Float] and [PredictionTypes::Usize] respectively.
pub fn predict(
    model: SupportedModels,
    data: Vec<f64>,
    probability: bool,
) -> PolarsResult<DataFrame> {
    let sh = (1, data.len());
    let sample = to_ndarray!(sh, data);
    let prediction = match model {
        SupportedModels::ElasticNet(m) => Some(PredictionTypes::Float(m.predict(sample))),
        SupportedModels::GaussianNaiveBayes(m) => Some(PredictionTypes::Usize(m.predict(sample))),
        SupportedModels::KMeans(m) => Some(PredictionTypes::Usize(m.predict(sample))),
        SupportedModels::LinearRegression(m) => Some(PredictionTypes::Float(m.predict(sample))),
        SupportedModels::LogisticRegression(m) => {
            if probability {
                Some(PredictionTypes::Probability(
                    m.predict_probabilities(&sample),
                ))
            } else {
                Some(PredictionTypes::Usize(m.predict(sample)))
            }
        }
        SupportedModels::DecisionTree(m) => Some(PredictionTypes::Usize(m.predict(sample))),
    };

    let prediction: DataFrame = match prediction {
        Some(v) => match v {
            PredictionTypes::Usize(m) => {
                let targets = m.targets.to_vec();
                vec_u64_to_df(targets, "prediction")?
            }
            PredictionTypes::Float(m) => {
                let targets = m.targets.to_vec();
                vec_f64_to_df(targets, "prediction")?
            }
            PredictionTypes::Probability(m) => vec_f64_to_df(m.to_vec(), "prediction")?,
        },
        None => {
            return PolarsResult::Err(PolarsError::ComputeError(polars::error::ErrString::Owned(
                "Failed to predict".to_string(),
            )))
        }
    };

    Ok(prediction)
}

pub fn inner_cross_validate(
    model: &SupportedModels,
    test_set: (DataFrame, DataFrame),
) -> Result<DataFrame, Status> {
    let result = match model {
        SupportedModels::LinearRegression(m) => (),
        SupportedModels::LogisticRegression(m) => (),
        SupportedModels::ElasticNet(m) => (),
        SupportedModels::DecisionTree(m) => (),
        SupportedModels::KMeans(m) => (),
        SupportedModels::GaussianNaiveBayes(m) => (),
    };

    let df = to_status_error(df!("scores" => &[0]));
    df
}
