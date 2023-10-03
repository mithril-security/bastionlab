use std::{error::Error, io::Cursor, sync::Mutex};

use ndarray::{prelude::*, Data, IxDynImpl, OwnedRepr, RawData};
use polars::{
    export::{
        arrow::types::PrimitiveType,
        num::{FromPrimitive, ToPrimitive},
        rayon::prelude::ParallelIterator,
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};
use tch::{kind::Element, CModule, Tensor};
use tokenizers::{FromPretrainedParameters, PaddingParams, Tokenizer, TruncationParams};
use tonic::Status;

// Update:
// - Adds macro to simplify converting Ndarrays into DataFrames

/// This macro simplifies converting an Ndarray ArrayBase<T> into a DataFrame
macro_rules! df_from_array {
    ($slice:ident, $to_fn:tt, $series_ty:tt, $col_name:expr) => {{
        let mut lanes: Vec<Series> = vec![];

        let d = $slice
            .into_iter()
            .map(|v| v.$to_fn().unwrap())
            .collect::<Vec<_>>();
        let series = Series::from($series_ty::new_vec($col_name, d));
        lanes.push(series);
        lanes
    }};
}

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
                "Unsupported data type in series: {}",
                d
            )))
        }
    })
}

pub fn ndarray_to_tensor<T: RawData, D: Dimension>(
    data: ArrayBase<T, D>,
) -> Result<tch::Tensor, Status>
where
    T: Data,
    T::Elem: Element,
{
    let tensor = Tensor::try_from(data)
        .map_err(|e| Status::aborted(format!("Could not convert ArrayBase to Tensor: {}", e)))?;

    Ok(tensor)
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
                "Unsupported data type in udf: {}",
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

fn get_tokenizer(
    model: &str,
    config: &str,
    revision: &str,
    auth_token: &str,
) -> Result<Tokenizer, Status> {
    let config: TokenizerParams = serde_json::from_str(config).map_err(|e| {
        Status::failed_precondition(format!(
            "Could not deserialize configuration for Tokenizer: {e}"
        ))
    })?;

    let mut tokenizer: Tokenizer = Tokenizer::from_pretrained(
        model,
        Some(FromPretrainedParameters {
            revision: revision.to_string(),
            auth_token: Some(auth_token.to_string()),
            ..Default::default()
        }),
    )
    .map_err(|_| Status::invalid_argument("Could not deserialize Hugging Face Tokenizer"))?;

    tokenizer.with_padding(config.padding_params);
    tokenizer.with_truncation(config.truncation_params);
    Ok(tokenizer)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerParams {
    padding_params: Option<PaddingParams>,
    truncation_params: Option<TruncationParams>,
}

pub fn list_to_ndarray<T>(v: Vec<T>, shape: Vec<usize>) -> Result<Array<T, Dim<IxDynImpl>>, Status>
where
    T: Clone,
{
    let arr = Array::from_shape_vec(shape, v)
        .map_err(|e| Status::aborted(format!("{e}")))?
        .as_standard_layout()
        .to_owned();

    Ok(arr)
}

pub fn series_to_tokenized_arrays<'a>(
    s: &Series,
    model: &str,
    config: &str,
    add_special_tokens: bool,
    revision: &str,
    auth_token: &str,
) -> Result<(Array<i64, Dim<IxDynImpl>>, Array<i64, Dim<IxDynImpl>>), Status> {
    let tokenizer = get_tokenizer(model, config, revision, auth_token)?;

    let out_ids = Arc::new(Mutex::new(Vec::new()));
    let ids_shape = Arc::new(Mutex::new(Vec::new()));
    let out_masks = Arc::new(Mutex::new(Vec::new()));
    let masks_shape = Arc::new(Mutex::new(Vec::new()));

    // We use the `par_iter` for parallel processing of the rows since there are no dependence.
    s.utf8().unwrap().par_iter().for_each(|row| match row {
        Some(s) => {
            let encoded = tokenizer
                .encode(s, add_special_tokens)
                .map_err(|e| Status::aborted(format!("Failed to tokenize string: {e}")))
                .unwrap();
            let mut ids = encoded
                .get_ids()
                .to_vec()
                .iter()
                .map(|v| *v as i64)
                .collect::<Vec<_>>();
            let mut mask = encoded
                .get_attention_mask()
                .to_vec()
                .iter()
                .map(|v| *v as i64)
                .collect::<Vec<_>>();

            ids_shape.lock().unwrap().push(ids.len());
            masks_shape.lock().unwrap().push(mask.len());
            out_ids.lock().unwrap().append(&mut ids);
            out_masks.lock().unwrap().append(&mut mask);
        }
        None => (),
    });

    let get_shape = |shape_vec: &[usize]| vec![shape_vec.len(), shape_vec[0]];
    let out_ids = out_ids.lock().unwrap().to_vec();
    let ids_shape = ids_shape.lock().unwrap().to_vec();
    let masks_shape = masks_shape.lock().unwrap().to_vec();
    let out_masks = out_masks.lock().unwrap().to_vec();

    let ids_shape = get_shape(&ids_shape);
    let masks_shape = get_shape(&masks_shape[..]);

    let out_ids = list_to_ndarray(out_ids, ids_shape)?;
    let out_masks = list_to_ndarray(out_masks, masks_shape)?;

    Ok((out_ids, out_masks))
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

/// This conversion is proprietary and wasn't completed by Polars.
pub fn ndarray_to_df<T, D: Dimension>(
    arr: &ArrayBase<OwnedRepr<T>, D>,
    col_names: Vec<String>,
) -> Result<DataFrame, Status>
where
    T: NumericNative + FromPrimitive + ToPrimitive,
{
    let mut lanes = vec![];

    for (i, col) in arr.columns().into_iter().enumerate() {
        // We convert the 1-d column into a standard layout in order to return it as a slice.
        let col = col.as_standard_layout();
        let col_name = &col_names[i];
        let mut vec_series = match col.as_slice() {
            Some(d) => {
                let vec_series = match T::PRIMITIVE {
                    PrimitiveType::Float64 => df_from_array!(d, to_f64, Float64Chunked, col_name),
                    PrimitiveType::Float32 => df_from_array!(d, to_f32, Float32Chunked, col_name),
                    PrimitiveType::Int64 => df_from_array!(d, to_i64, Int64Chunked, col_name),
                    PrimitiveType::Int32 => df_from_array!(d, to_i32, Int32Chunked, col_name),
                    PrimitiveType::UInt64 => df_from_array!(d, to_u64, UInt64Chunked, col_name),
                    PrimitiveType::UInt32 => df_from_array!(d, to_u32, UInt32Chunked, col_name),
                    _ => {
                        return Err(Status::aborted(format!(
                            "Conversion from {:?} into DataFrame not yet supported",
                            T::PRIMITIVE
                        )))
                    }
                };
                vec_series
            }
            None => {
                return Err(Status::aborted(
                    "Could not convert column in array to slice",
                ));
            }
        };

        lanes.append(&mut vec_series);
    }

    DataFrame::new(lanes).map_err(|e| Status::aborted(format!("Could not create a dataframe: {e}")))
}
