use polars::prelude::*;
use tch::{kind::Element, Tensor};
use tonic::Status;

pub fn sanitize_df(df: &mut DataFrame, blacklist: &Vec<String>) {
    for name in blacklist {
        let idx = match df.get_column_names().iter().position(|x| x == name) {
            Some(idx) => idx,
            None => continue,
        };
        let series = df.get_columns_mut().get_mut(idx).unwrap();
        *series = Series::full_null(name, series.len(), series.dtype());
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
