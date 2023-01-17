use std::{error::Error, io::Cursor};

use bastionlab_common::utils::array_to_tensor;
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use ndarray::{ArrayBase, CowRepr, Data, Dimension, Ix2, RawData};
use polars::{
    export::ahash::HashSet,
    prelude::{
        row::{AnyValueBuffer, Row},
        *,
    },
};
use regex::Regex;
use tch::{kind::Element, CModule, Tensor};
use tokenizers::{Encoding, Tokenizer};
use tonic::Status;

use crate::composite_plan::StringMethod;

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
        DataType::Float32 => array_to_tensor(series.f32().unwrap())?,
        DataType::Float64 => array_to_tensor(series.f64().unwrap())?,
        DataType::Int64 => array_to_tensor(series.i64().unwrap())?,
        DataType::Int32 => array_to_tensor(series.i32().unwrap())?,
        DataType::Int16 => array_to_tensor(series.i16().unwrap())?,
        DataType::Int8 => array_to_tensor(series.i8().unwrap())?,
        DataType::UInt32 => {
            let s = to_status_error(series.cast(&DataType::Int64))?;
            array_to_tensor(s.i64().unwrap())?
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

#[allow(unused)]
pub fn tokenized_series_to_series(vs: Vec<Series>, model: &str) -> Result<Series, Status> {
    let tokenizer = get_tokenizer(model)?;

    let get_list = |v: AnyValue| match v {
        AnyValue::List(s) => Some(s),
        _ => None,
    };
    for idx in 0..vs[0].len() {
        let id_tokens = vs[0].get(idx);
        match get_list(id_tokens) {
            Some(v) => {
                println!("{:?}", v);
            }
            None => (),
        }
        let mask_tokens = vs[1].get(idx);
        match get_list(mask_tokens) {
            Some(v) => {
                println!("{:?}", v);
            }
            None => (),
        }
    }

    Ok(Series::new_empty("", &DataType::Utf8))
}

fn to_series(v: &[AnyValue]) -> Result<Series, Status> {
    let s = Series::from_any_values("col", v)
        .map_err(|e| Status::aborted(format!("Failed to create Series from AnyValues: {e}")))?;
    Ok(s)
}

pub fn apply_method(method: &StringMethod, series: &Series) -> Result<Series, Status> {
    let StringMethod { name, pattern, to } = method;

    let get_inner_opt = |v: &Option<String>| -> Result<String, Status> {
        match v {
            Some(v) => Ok(v.clone()),
            None => {
                return Err(Status::failed_precondition(format!(
                    "{:?} necessary for {name}",
                    v
                )));
            }
        }
    };

    let create_regex = |pat: &str| -> Result<Regex, Status> {
        Regex::new(pat)
            .map_err(|e| Status::aborted(format!("Could not create regex from {pat}: {e}")))
    };
    let series = series.utf8().unwrap();
    let output = match name.as_str() {
        "split" => {
            let pat = get_inner_opt(pattern)?;
            let mut out = vec![];
            for elem in series.into_iter() {
                let value = match elem {
                    Some(r) => {
                        let s = r.split(&pat).into_vec();
                        let v =
                            AnyValue::List(Utf8Chunked::from_slice("col", &s[..]).into_series());
                        v
                    }
                    None => {
                        return Err(Status::aborted(format!("Could not apply split to null")));
                    }
                };
                out.push(value);
            }
            let s = to_series(&out[..])?;
            s
        }
        "to_lowercase" => series.to_lowercase().into_series(),
        "to_uppercase" => series.to_uppercase().into_series(),
        "replace" => {
            let pat = get_inner_opt(pattern)?;
            let to = get_inner_opt(to)?;
            let re = create_regex(&pat)?;

            series
                .replace(&re.as_str(), &to)
                .map_err(|e| {
                    Status::aborted(format!("Failed to replace {pat} with {to} for {name}: {e}"))
                })?
                .into_series()
        }
        "replace_all" => {
            let pat = get_inner_opt(pattern)?;
            let to = get_inner_opt(to)?;
            let re = create_regex(&pat)?;

            series
                .replace_all(&re.as_str(), &to)
                .map_err(|e| {
                    Status::aborted(format!(
                        "Failed to replace all {pat} with {to} for {name}: {e}"
                    ))
                })?
                .into_series()
        }
        "contains" => {
            let pat = get_inner_opt(pattern)?;
            let re = create_regex(&pat)?;

            series
                .contains(&re.as_str())
                .map_err(|e| Status::aborted(format!("Could not find a match: {e}")))?
                .into_series()
        }
        "match" => {
            let pat = get_inner_opt(pattern)?;
            let mut out = Vec::with_capacity(series.len());

            let re = create_regex(&pat)?;

            for elem in series.into_iter() {
                let value = match elem {
                    Some(r) => {
                        let s = if re.find_iter(r).last().is_none() {
                            false
                        } else {
                            true
                        };

                        AnyValue::Boolean(s)
                    }
                    None => {
                        return Err(Status::aborted(format!("Could not apply split to null")));
                    }
                };

                out.push(value);
            }

            let s = to_series(&out[..])?;
            s
        }
        "findall" => {
            let pat = get_inner_opt(pattern)?;
            let mut out = Vec::with_capacity(series.len());

            let re = create_regex(&pat)?;
            for elem in series.into_iter() {
                let value = match elem {
                    Some(r) => {
                        let s = re
                            .find_iter(r)
                            .map(|s| s.as_str().to_owned())
                            .collect::<Vec<_>>();

                        let s = Utf8Chunked::from_slice("", &s[..]);
                        let s = if !s.is_empty() {
                            s
                        } else {
                            let s: Vec<String> = vec![];
                            Utf8Chunked::from_slice("", &s[..])
                        };

                        AnyValue::List(s.into_series())
                    }
                    None => {
                        return Err(Status::aborted(format!("Could not apply split to null")));
                    }
                };

                out.push(value);
            }
            let s = to_series(&out[..])?;
            s
        }
        "extract" => {
            let pat = get_inner_opt(pattern)?;
            let re = create_regex(&pat)?;
            series
                .extract_all(&re.as_str())
                .map_err(|e| Status::aborted(format!("Could not execute extract on Series: {e}")))?
                .into_series()
        }
        "fuzzy_match" => {
            let pat = get_inner_opt(pattern)?;
            let m = SkimMatcherV2::default();
            let mut out = vec![];
            for elem in series.into_iter() {
                let value = match elem {
                    Some(s) => {
                        let v = m.fuzzy_indices(s, &pat);
                        match v {
                            Some(_) => AnyValue::Utf8(s),
                            None => AnyValue::Null,
                        }
                    }
                    None => {
                        return Err(Status::aborted(format!("Could not apply split to null")));
                    }
                };

                out.push(value);
            }

            let s = to_series(&out[..])?;
            s
        }
        _ => {
            return Err(Status::invalid_argument(format!("{name} is unsupported")));
        }
    };
    Ok(output)
}
