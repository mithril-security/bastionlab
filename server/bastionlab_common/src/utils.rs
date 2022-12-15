use polars::{
    prelude::{
        ChunkedArray, DataType, Float32Type, Int16Type, Int32Type, Int64Type, Int8Type,
        PolarsNumericType,
    },
    series::Series,
};
use tch::{kind::Element, Kind, Tensor};
use tonic::Status;

pub fn tensor_to_series(name: &str, dtype: &DataType, tensor: Tensor) -> Result<Series, Status> {
    Ok(match dtype {
        DataType::Float32 => Series::from(tensor_to_array::<Float32Type>(&name, tensor)),
        DataType::Float64 => Series::from(tensor_to_array::<Float32Type>(&name, tensor)),
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

pub fn kind_to_datatype(kind: Kind) -> DataType {
    match kind {
        Kind::Uint8 => DataType::UInt8,
        Kind::Int8 => DataType::Int8,
        Kind::Int16 => DataType::Int16,
        Kind::Int => DataType::Int32,
        Kind::Int64 => DataType::Int64,
        Kind::Half => DataType::Float32,
        Kind::Float => DataType::Float32,
        Kind::Double => DataType::Float64,
        Kind::Bool => DataType::Boolean,
        _ => DataType::Unknown,
    }
}
