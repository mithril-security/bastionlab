use bastionlab_common::{
    common_conversions::{series_to_tensor, tensor_to_series2},
    prelude::*,
};
use polars::lazy::dsl::{Context, *};
use polars::prelude::*;
use serde::{de::Visitor, Deserialize, Deserializer, Serialize, Serializer};

use crate::composite_plan::CompositePlan;

struct TorchUdfDeserializer;

impl UdfDeserializer for TorchUdfDeserializer {
    fn deserialize_udf(
        &self,
        deserializer: &mut dyn erased_serde::Deserializer,
    ) -> Result<Arc<dyn SerializableUdf>, erased_serde::Error> {
        Ok(Arc::new(UdfLambda::deserialize(deserializer)?) as _)
    }
}

#[derive(Debug)]
pub struct TorchFunc(tch::CModule);

impl TorchFunc {
    fn call_unary(&self, series: &Series, name: &str) -> PolarsResult<Series> {
        let tensor = series_to_tensor(series)?;
        let ret = self
            .0
            .forward_ts(&[&tensor])
            .map_err(|err| PolarsError::ComputeError(format!("{err}").into()))?;
        tensor_to_series2(name, ret)
    }

    fn call_binary(&self, a: &Series, b: &Series, name: &str) -> PolarsResult<Series> {
        let a = series_to_tensor(a)?;
        let b = series_to_tensor(b)?;
        let ret = self
            .0
            .forward_ts(&[&a, &b])
            .map_err(|err| PolarsError::ComputeError(format!("{err}").into()))?;
        tensor_to_series2(name, ret)
    }

    fn call_slice(&self, series: &[Series], name: &str) -> PolarsResult<Series> {
        let series = series
            .iter()
            .map(|el| series_to_tensor(el))
            .collect::<Result<Vec<tch::Tensor>, _>>()?;
        let ret = self
            .0
            .forward_ts(&series)
            .map_err(|err| PolarsError::ComputeError(format!("{err}").into()))?;
        tensor_to_series2(name, ret)
    }
}

impl Serialize for TorchFunc {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // TODO: tch does not implement load from buffer
        // this needs to be fixed upstream
        "<user-defined function>".serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TorchFunc {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = TorchFunc;

            fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "a torchscript module")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                let mut reader = v.as_bytes();
                let mut reader = base64::read::DecoderReader::new(&mut reader, base64::STANDARD);

                let cmodule = tch::CModule::load_data(&mut reader).map_err(|e| {
                    serde::de::Error::custom(format!("Error loading torch module: {e}"))
                })?;
                Ok(TorchFunc(cmodule))
            }
        }
        deserializer.deserialize_str(V)
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum UdfLambdaOp {
    // fn(a) -> r, flat
    MapSingle {
        agg_list: bool,
    },
    // fn(...args) -> r, flat or groups
    MapMultiple {
        apply_groups: bool,
        returns_scalar: bool,
    },
    // fn(a, b) -> r, groups
    Fold,
    Reduce,
    CumFold {
        include_init: bool,
    },
    CumReduce,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct UdfLambda {
    lambda: TorchFunc,
    output_type: Option<DataType>,
    op: UdfLambdaOp,
}

impl SerializableUdf for UdfLambda {
    fn as_debug(&self) -> &dyn std::fmt::Debug {
        self
    }
    fn as_serialize(&self) -> Option<&dyn erased_serde::Serialize> {
        Some(self)
    }

    fn call_series_slice(&self, series: &mut [Series]) -> PolarsResult<Series> {
        let output_type = self.output_type.as_ref().unwrap_or(&DataType::Unknown);
        let res = match &self.op {
            UdfLambdaOp::MapSingle { .. } => {
                let series = &series[0];

                self.lambda.call_unary(series, series.name())?
            }
            UdfLambdaOp::MapMultiple { .. } => self.lambda.call_slice(series, "")?,
            UdfLambdaOp::Fold => {
                let mut series = series.to_vec();
                let mut acc = series.pop().unwrap(); // last argument is the accumulator

                for s in series {
                    acc = self.lambda.call_binary(&acc, &s, acc.name())?;
                }
                acc
            }
            UdfLambdaOp::Reduce => {
                let mut s = series.to_vec();
                let mut s_iter = s.drain(..);

                match s_iter.next() {
                    Some(mut acc) => {
                        for s in s_iter {
                            acc = self.lambda.call_binary(&acc, &s, acc.name())?;
                        }
                        acc
                    }
                    None => {
                        return Err(PolarsError::ComputeError(
                            "Reduce did not have any expressions to fold".into(),
                        ))
                    }
                }
            }
            UdfLambdaOp::CumFold { include_init } => {
                let mut series = series.to_vec();
                let mut acc = series.pop().unwrap();

                let mut result = vec![];
                if *include_init {
                    result.push(acc.clone())
                }

                for s in series {
                    let name = s.name().to_string();
                    acc = self.lambda.call_binary(&acc, &s, acc.name())?;
                    acc.rename(&name);
                    result.push(acc.clone());
                }

                StructChunked::new(acc.name(), &result).map(|ca| ca.into_series())?
            }
            UdfLambdaOp::CumReduce => {
                let mut s = series.to_vec();
                let mut s_iter = s.drain(..);

                match s_iter.next() {
                    Some(mut acc) => {
                        let mut result = vec![acc.clone()];

                        for s in s_iter {
                            let name = s.name().to_string();
                            acc = self.lambda.call_binary(&acc, &s, acc.name())?;
                            acc.rename(&name);
                            result.push(acc.clone());
                        }

                        StructChunked::new(acc.name(), &result).map(|ca| ca.into_series())?
                    }
                    None => {
                        return Err(PolarsError::ComputeError(
                            "Reduce did not have any expressions to fold".into(),
                        ))
                    }
                }
            }
        };

        if !matches!(output_type, DataType::Unknown) && res.dtype() != output_type {
            Err(PolarsError::SchemaMisMatch(
                    format!("Expected output type: '{:?}', but got '{:?}'. Set 'return_dtype' to the proper datatype.", output_type, res.dtype()).into()))
        } else {
            Ok(res)
        }
    }

    fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        match &self.op {
            UdfLambdaOp::MapSingle { .. } => {
                get_output::map_field(move |fld| match self.output_type {
                    Some(ref dt) => Field::new(fld.name(), dt.clone()),
                    None => {
                        let mut fld = fld.clone();
                        fld.coerce(DataType::Unknown);
                        fld
                    }
                })(fields)
            }
            UdfLambdaOp::MapMultiple { .. } => {
                get_output::map_field(move |fld| match self.output_type {
                    Some(ref dt) => Field::new(fld.name(), dt.clone()),
                    None => fld.clone(),
                })(fields)
            }
            UdfLambdaOp::Fold | UdfLambdaOp::Reduce => get_output::super_type()(fields),
            UdfLambdaOp::CumFold { .. } | UdfLambdaOp::CumReduce => {
                let st = get_output::super_type()(fields)?.dtype;
                Ok(Field::new(
                    &fields[0].name,
                    DataType::Struct(
                        fields
                            .iter()
                            .map(|fld| Field::new(fld.name(), st.clone()))
                            .collect(),
                    ),
                ))
            }
        }
    }
}

pub fn deserialize_composite_plan(s: &str) -> Result<CompositePlan, serde_json::Error> {
    let _ = set_udf_deserializer(Box::new(TorchUdfDeserializer));
    serde_json::from_str(s)
}
