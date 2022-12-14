use std::{error::Error, i64::MAX, sync::Mutex};

use polars::prelude::*;
use tch::{kind::Element, Tensor};
use tonic::Status;

use crate::polars_proto::meta::Shape;

pub fn sanitize_df(df: &mut DataFrame, blacklist: &Vec<String>) {
    for name in blacklist {
        let idx = match df.get_column_names().iter().position(|x| x == name) {
            Some(idx) => idx,
            None => continue,
        };
        let series = df.get_columns_mut().get_mut(idx).unwrap();
        *series = Series::new_empty(name, series.dtype());
    }
}

pub fn series_to_tensor(series: &Series) -> Result<Tensor, Status> {
    Ok(match series.dtype() {
        DataType::Float32 => array_to_tensor(series.f32().unwrap())?,
        DataType::Float64 => array_to_tensor(series.f64().unwrap())?,
        DataType::Int64 => array_to_tensor(series.i64().unwrap())?,
        DataType::Int32 => array_to_tensor(series.i32().unwrap())?,
        DataType::Int16 => array_to_tensor(series.i16().unwrap())?,
        DataType::Int8 => array_to_tensor(series.i8().unwrap())?,
        d => {
            return Err(Status::invalid_argument(format!(
                "Unsuported data type in udf: {}",
                d
            )))
        }
    })
}

pub fn vec_series_to_tensor(
    v_series: Vec<&Series>,
) -> Result<(Vec<Mutex<Tensor>>, Vec<Shape>, Vec<String>, i64), Status> {
    let mut ts = Vec::new();
    let mut shapes = Vec::new();
    let mut dtypes = Vec::new();
    let mut nb_samples = MAX;
    for s in v_series {
        let t = series_to_tensor(s)?;
        if nb_samples == MAX {
            nb_samples = t.size()[0];
        }
        shapes.push(Shape { elem: t.size() });
        dtypes.push(format!("{:?}", t.kind()));
        ts.push(Mutex::new(t));
    }

    Ok((ts, shapes, dtypes, nb_samples))
}

pub fn tensor_to_series(name: &str, dtype: &DataType, tensor: Tensor) -> Result<Series, Status> {
    Ok(match dtype {
        DataType::Float32 => Series::from(tensor_to_array::<Float32Type>(&name, tensor)),
        DataType::Float64 => Series::from(tensor_to_array::<Float64Type>(&name, tensor)),
        DataType::Int64 => Series::from(tensor_to_array::<Int64Type>(&name, tensor)),
        DataType::Int32 => Series::from(tensor_to_array::<Int32Type>(&name, tensor)),
        DataType::Int16 => Series::from(tensor_to_array::<Int16Type>(&name, tensor)),
        DataType::Int8 => Series::from(tensor_to_array::<Int8Type>(&name, tensor)),
        d => {
            return Err(Status::invalid_argument(format!(
                "Unsuported data type in udf: {}",
                d
            )))
        }
    })
}

pub fn array_to_tensor<T>(series: &ChunkedArray<T>) -> Result<Tensor, Status>
where
    T: PolarsNumericType,
    T::Native: Element,
{
    Ok(match series.rechunk().cont_slice() {
        Ok(slice) => Tensor::from(slice),
        Err(_) => {
            if !series.has_validity() {
                return Err(Status::invalid_argument(
                    "Cannot apply udf on a column that contains empty values",
                ));
            }
            let v: Vec<T::Native> = series.into_no_null_iter().collect();
            Tensor::from(&v[..])
        }
    })
}

pub fn tensor_to_array<T>(name: &str, tensor: Tensor) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Element,
{
    let v = Vec::from(tensor);
    ChunkedArray::new_vec(name, v)
}

pub fn lazy_frame_from_logical_plan(plan: LogicalPlan) -> LazyFrame {
    let mut ldf = LazyFrame::default();
    ldf.logical_plan = plan;
    ldf
}

pub fn to_status_error<T, E: Error>(input: Result<T, E>) -> Result<T, Status> {
    input.map_err(|err| Status::aborted(err.to_string()))
}
