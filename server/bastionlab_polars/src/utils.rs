use std::{error::Error, io::Cursor, ops::Add, sync::Mutex};

use bastionlab_common::utils::array_to_tensor;
use polars::prelude::{
    row::{AnyValueBuffer, Row},
    *,
};
use tch::{CModule, Tensor};
use tokenizers::{Encoding, Tokenizer};
use tonic::Status;

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

            println!("{:?}", shape);
            let out = list_dtype_to_tensor(series)?;
            let mut zeroes = Tensor::zeros(&shape[..], (out[0].kind(), out[0].device()));
            for t in out.iter() {
                zeroes = zeroes.add(t);
            }
            zeroes.print();
            zeroes
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
) -> Result<(Vec<Mutex<Tensor>>, Vec<i64>, Vec<String>, i32), Status> {
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
        ts.push(Mutex::new(t.data()));
    }
    Ok((ts, shapes, dtypes, nb_samples.try_into().unwrap()))
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
