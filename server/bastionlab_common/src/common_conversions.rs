use std::{error::Error, io::Cursor};

use ndarray::{prelude::*, CowRepr, Data, RawData};
use polars::{
    export::ahash::HashSet,
    prelude::{
        row::{AnyValueBuffer, Row},
        *,
    },
};
use tch::{kind::Element, CModule, Tensor};
use tokenizers::{Encoding, Tokenizer};
use tonic::Status;

pub fn list_dtype_to_tensor(series: &Series) -> Result<Vec<Tensor>, Status> {
    let rows = to_status_error(series.list())?;
    let mut out = vec![];
    for s in rows.into_iter() {
        match s.as_ref() {
            Some(s) => out.push(series_to_tensor(s)?),
            None => return Err(Status::aborted("Could not iterate over series.")),
        }
    }

    Ok(out)
}
pub fn series_to_tensor(series: &Series) -> Result<Tensor, Status> {
    Ok(match series.dtype() {
        DataType::Float32 => chunked_array_to_tensor(series.f32().unwrap())?,
        DataType::Float64 => chunked_array_to_tensor(series.f64().unwrap())?,
        DataType::Int64 => chunked_array_to_tensor(series.i64().unwrap())?,
        DataType::Int32 => chunked_array_to_tensor(series.i32().unwrap())?,
        DataType::Int16 => chunked_array_to_tensor(series.i16().unwrap())?,
        DataType::Int8 => chunked_array_to_tensor(series.i8().unwrap())?,
        DataType::UInt32 => {
            let s = to_status_error(series.cast(&DataType::Int64))?;
            chunked_array_to_tensor(s.i64().unwrap())?
        }
        DataType::List(_) => {
            let mut shape = vec![];
            let first = series.get(0);
            shape.push(series.len() as i64);
            if let AnyValue::List(l) = first {
                shape.push(l.len() as i64);
            };

            let out = list_dtype_to_tensor(series)?;
            let mut zeros = Tensor::zeros(&shape[..], (out[0].kind(), out[0].device()));

            for (i, t) in out.iter().enumerate() {
                let index = Tensor::from(i as i64);
                zeros = zeros.index_put(&vec![Some(index.copy())][..], t, false);
            }

            zeros
        }
        d => {
            return Err(Status::invalid_argument(format!(
                "Unsuported data type in series: {}",
                d
            )))
        }
    })
}

pub fn vec_series_to_tensor(
    v_series: Vec<&Series>,
) -> Result<(Vec<Tensor>, Vec<i64>, Vec<String>, i32), Status> {
    let mut ts = Vec::new();
    let mut shapes = Vec::new();
    let mut dtypes = Vec::new();
    let nb_samples = match v_series.first() {
        Some(v) => v.len(),
        None => 0,
    };
    for s in v_series {
        let t = series_to_tensor(s)?;
        shapes.push(t.size()[1]);
        dtypes.push(format!("{:?}", t.kind()));
        ts.push(t);
    }
    Ok((ts, shapes, dtypes, nb_samples.try_into().unwrap()))
}

fn ndarray_to_tensor<T: RawData, D: Dimension>(
    data: &ArrayBase<T, D>,
) -> Result<tch::Tensor, Status>
where
    T: Data,
    T::Elem: Element,
{
    let tensor = Tensor::try_from(data)
        .map_err(|e| Status::aborted(format!("Could not convert ArrayBase to Tensor: {}", e)))?;

    Ok(tensor)
}

pub fn df_to_tensor(df: &DataFrame) -> Result<Tensor, Status> {
    // Make sure all the dtypes are same.
    let set = HashSet::from_iter(df.dtypes().iter().map(|dtype| dtype.to_string()));
    if set.len() > 1 {
        return Err(Status::aborted(
            "DataTypes for all columns should be the same",
        ));
    }

    let dtype = &df.dtypes()[0];

    match dtype {
        DataType::Float32 => ndarray_to_tensor::<CowRepr<f32>, Ix2>(
            &df.to_ndarray::<Float32Type>()
                .map_err(|e| {
                    Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                })?
                .as_standard_layout(),
        ),
        DataType::Float64 => ndarray_to_tensor::<CowRepr<f64>, Ix2>(
            &df.to_ndarray::<Float64Type>()
                .map_err(|e| {
                    Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                })?
                .as_standard_layout(),
        ),
        DataType::Int64 => ndarray_to_tensor::<CowRepr<i64>, Ix2>(
            &df.to_ndarray::<Int64Type>()
                .map_err(|e| {
                    Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                })?
                .as_standard_layout(),
        ),
        DataType::Int32 => ndarray_to_tensor::<CowRepr<i32>, Ix2>(
            &df.to_ndarray::<Int32Type>()
                .map_err(|e| {
                    Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                })?
                .as_standard_layout(),
        ),
        DataType::Int16 => ndarray_to_tensor::<CowRepr<i16>, Ix2>(
            &df.to_ndarray::<Int16Type>()
                .map_err(|e| {
                    Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                })?
                .as_standard_layout(),
        ),
        DataType::Int8 => ndarray_to_tensor::<CowRepr<i8>, Ix2>(
            &df.to_ndarray::<Int8Type>()
                .map_err(|e| {
                    Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                })?
                .as_standard_layout(),
        ),
        _ => {
            return Err(Status::aborted(format!("Unsupported datatype {}", dtype)));
        }
    }
}

pub fn tensor_to_series(name: &str, dtype: &DataType, tensor: Tensor) -> Result<Series, Status> {
    Ok(match dtype {
        DataType::Float32 => Series::from(tensor_to_chunked_array::<Float32Type>(&name, tensor)),
        DataType::Float64 => Series::from(tensor_to_chunked_array::<Float64Type>(&name, tensor)),
        DataType::Int64 => Series::from(tensor_to_chunked_array::<Int64Type>(&name, tensor)),
        DataType::Int32 => Series::from(tensor_to_chunked_array::<Int32Type>(&name, tensor)),
        DataType::Int16 => Series::from(tensor_to_chunked_array::<Int16Type>(&name, tensor)),
        DataType::Int8 => Series::from(tensor_to_chunked_array::<Int8Type>(&name, tensor)),
        d => {
            return Err(Status::invalid_argument(format!(
                "Unsuported data type in udf: {}",
                d
            )))
        }
    })
}

pub fn lazy_frame_from_logical_plan(plan: LogicalPlan) -> LazyFrame {
    let mut ldf = LazyFrame::default();
    ldf.logical_plan = plan;
    ldf
}

pub fn to_status_error<T, E: Error>(input: Result<T, E>) -> Result<T, Status> {
    input.map_err(|err| Status::aborted(err.to_string()))
}

pub fn load_udf(udf: String) -> Result<CModule, Status> {
    Ok(
        CModule::load_data(&mut Cursor::new(base64::decode(udf).map_err(|e| {
            Status::invalid_argument(format!("Could not decode bas64-encoded udf: {}", e))
        })?))
        .map_err(|e| {
            Status::invalid_argument(format!("Could not deserialize udf from bytes: {}", e))
        })?,
    )
}

fn get_tokenizer(model: &str) -> Result<Tokenizer, Status> {
    let model = base64::decode_config(model, base64::STANDARD).map_err(|e| {
        Status::invalid_argument(format!("Could not decode bas64-encoded udf: {}", e))
    })?;
    let tokenizer: Tokenizer = Tokenizer::from_bytes(model)
        .map_err(|_| Status::invalid_argument("Could not deserialize Hugging Face Tokenizer"))?;
    Ok(tokenizer)
}

pub fn series_to_tokenized_series(
    s: &Series,
    name: &str,
    model: &str,
) -> Result<DataFrame, Status> {
    let tokenizer = get_tokenizer(model)?;
    let mut batched_seqs = Vec::new();

    let to_row = |tokens: &Encoding| {
        let ids = tokens.get_ids();
        let mask = tokens.get_attention_mask();

        let to_any_value = |v: &[u32]| -> AnyValue {
            let mut buf = AnyValueBuffer::new(&DataType::UInt32, v.len());
            v.iter().for_each(|v| {
                buf.add(AnyValue::UInt32(*v));
            });
            AnyValue::List(buf.into_series())
        };
        let ids = to_any_value(ids);
        let mask = to_any_value(mask);

        let joined = vec![ids.clone(), mask.clone()];

        let row = Row::new(joined);
        row
    };
    for row in s.utf8().unwrap().into_iter() {
        match row {
            Some(s) => {
                batched_seqs.push(s.to_string());
            }
            None => {
                return Err(Status::aborted(
                    "Failed to convert row to Utf8 string".to_string(),
                ));
            }
        }
    }

    let tokens_vec = tokenizer
        .encode_batch(batched_seqs, false)
        .map_err(|_| Status::aborted("Failed to tokenize string"))?;

    let rows = tokens_vec.iter().map(to_row).collect::<Vec<_>>();
    let mut df = to_status_error(DataFrame::from_rows(&rows[..]))?;

    let ids_names = &format!("{}_ids", name.to_lowercase());
    let mask_names = &format!("{}_mask", name.to_lowercase());

    let col_names = df.get_column_names_owned();
    to_status_error(df.rename(&col_names[0], &ids_names))?;
    to_status_error(df.rename(&col_names[1], &mask_names))?;
    Ok(df)
}

pub fn chunked_array_to_tensor<T>(series: &ChunkedArray<T>) -> Result<Tensor, Status>
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

pub fn tensor_to_chunked_array<T>(name: &str, tensor: Tensor) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Element,
{
    let v = Vec::from(tensor);
    ChunkedArray::new_vec(name, v)
}
