use std::{error::Error, i64::MAX, sync::Mutex};

use bastionlab_common::utils::array_to_tensor;
use polars::prelude::*;
use tch::Tensor;
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
        let t = match s.dtype() {
            DataType::List(_) => {
                let mut out = list_dtype_to_tensor(s)?;
                for t in out.iter_mut() {
                    *t = t.unsqueeze(0);
                }
                Tensor::cat(&out[..], 0)
            }
            _ => series_to_tensor(s)?,
        };
        if nb_samples == MAX {
            nb_samples = t.size()[0];
        }
        shapes.push(Shape { elem: t.size() });
        dtypes.push(format!("{:?}", t.kind()));
        ts.push(Mutex::new(t));
    }

    Ok((ts, shapes, dtypes, nb_samples))
}

pub fn lazy_frame_from_logical_plan(plan: LogicalPlan) -> LazyFrame {
    let mut ldf = LazyFrame::default();
    ldf.logical_plan = plan;
    ldf
}

pub fn to_status_error<T, E: Error>(input: Result<T, E>) -> Result<T, Status> {
    input.map_err(|err| Status::aborted(err.to_string()))
}
