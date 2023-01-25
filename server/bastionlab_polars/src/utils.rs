use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use polars::prelude::*;
use regex::Regex;
use tch::{kind::Element, Tensor};
use tonic::Status;

use crate::composite_plan::StringMethod;

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
                        let s = r.split(&pat).collect::<Vec<_>>();
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
                .extract(&re.as_str(), 0)
                .map_err(|e| Status::aborted(format!("Could not execute extract on Series: {e}")))?
                .into_series()
        }
        "extract_all" => {
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

fn to_series(v: &[AnyValue]) -> Result<Series, Status> {
    let s = Series::from_any_values("col", v)
        .map_err(|e| Status::aborted(format!("Failed to create Series from AnyValues: {e}")))?;
    Ok(s)
}
