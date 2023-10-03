use std::sync::{Arc, Mutex};

use bastionlab_common::{array_store::ArrayStore, common_conversions::*};
use bastionlab_polars::BastionLabPolars;
use bastionlab_torch::BastionLabTorch;
use ndarray::{Axis, Dim, IxDynImpl, OwnedRepr};
use polars::export::ahash::HashSet;
use polars::prelude::*;
use tch::Tensor;
use tonic::{Request, Response, Status};

use crate::conversion_proto::{conversion_service_server::ConversionService, ToTokenizedArrays};
use crate::conversion_proto::{RemoteArray, RemoteArrays, RemoteDataFrame};

use crate::bastionlab::Reference;
pub struct Converter {
    torch: Arc<BastionLabTorch>,
    polars: Arc<BastionLabPolars>,
}

impl Converter {
    pub fn new(torch: Arc<BastionLabTorch>, polars: Arc<BastionLabPolars>) -> Self {
        Self { torch, polars }
    }
    pub fn ndarray_to_tensor(&self, arr: ArrayStore) -> Result<Tensor, Status> {
        let tensor = match arr {
            ArrayStore::AxdynI64(a) => ndarray_to_tensor::<OwnedRepr<i64>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynF64(a) => ndarray_to_tensor::<OwnedRepr<f64>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynF32(a) => ndarray_to_tensor::<OwnedRepr<f32>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynI32(a) => ndarray_to_tensor::<OwnedRepr<i32>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynI16(a) => ndarray_to_tensor::<OwnedRepr<i16>, Dim<IxDynImpl>>(a),
            _ => {
                return Err(Status::aborted(format!(
                    "Cannot convert {:?} into Tensor",
                    arr
                )))
            }
        };

        tensor
    }

    pub fn df_to_ndarray(&self, df: &DataFrame) -> Result<String, Status> {
        let set = HashSet::from_iter(df.dtypes().iter().map(|dtype| dtype.to_string()));
        if set.len() > 1 {
            return Err(Status::aborted(
                "DataTypes for all columns should be the same",
            ));
        }

        let dtype = &df.dtypes()[0];

        let arr = match dtype {
            DataType::Float32 => {
                let arr = df
                    .to_ndarray::<Float32Type>()
                    .map_err(|e| {
                        Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                    })?
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn();
                ArrayStore::AxdynF32(arr)
            }
            DataType::Float64 => {
                let arr = df
                    .to_ndarray::<Float64Type>()
                    .map_err(|e| {
                        Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                    })?
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn();
                ArrayStore::AxdynF64(arr)
            }

            DataType::Int64 => {
                let arr = df
                    .to_ndarray::<Int64Type>()
                    .map_err(|e| {
                        Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                    })?
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn();
                ArrayStore::AxdynI64(arr)
            }
            DataType::Int32 => {
                let arr = df
                    .to_ndarray::<Int32Type>()
                    .map_err(|e| {
                        Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                    })?
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn();
                ArrayStore::AxdynI32(arr)
            }
            DataType::UInt32 => {
                let arr = df
                    .to_ndarray::<UInt32Type>()
                    .map_err(|e| {
                        Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                    })?
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn();
                ArrayStore::AxdynU32(arr)
            }
            DataType::UInt64 => {
                let arr = df
                    .to_ndarray::<UInt64Type>()
                    .map_err(|e| {
                        Status::aborted(format!("Cound not convert DataFrame to ndarray: {}", e))
                    })?
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn();
                ArrayStore::AxdynU64(arr)
            }
            _ => {
                return Err(Status::aborted(format!("{:?} not support ", dtype)));
            }
        };
        let identifier = self.polars.insert_array(arr);
        Ok(identifier)
    }

    pub fn list_series_to_array_store(&self, series: &Series) -> Result<ArrayStore, Status>
where {
        let get_shape = |series: &Series| -> Option<Vec<usize>> {
            let _0th = series.len();
            let _1st = if let AnyValue::List(inner) = series.get(0) {
                inner.len()
            } else {
                return None;
            };
            Some(vec![_0th, _1st])
        };

        /*
           In this function, we are only working with list series types.

           The function serves as a helper function to convert series into a list to
           reconstruct into ArrayStore(ArrayBase<T>)
        */
        let dtype = series.dtype();

        let arraystore = match dtype {
            DataType::List(inner) => {
                /*
                   In Polars, we do not expect List[List[Primitive]].
                   Only List[Primitive] is allowed.
                */
                let res = match inner.as_ref() {
                    DataType::Float64 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.f64())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynF64(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }

                    DataType::Float32 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.f32())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynF32(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }

                    DataType::Int64 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.i64())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynI64(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }

                    DataType::Int32 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.i32())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynI32(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }

                    DataType::Int16 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.i16())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynI16(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }
                    DataType::UInt64 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.u64())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynU64(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }
                    DataType::UInt32 => {
                        let shape = get_shape(series).ok_or(Status::aborted(
                            "Only List Series are supported in get_shape",
                        ))?;
                        let exploded = to_status_error(series.explode())?;
                        let arr = to_status_error(exploded.u32())?;
                        let slice = to_status_error(arr.cont_slice())?;
                        ArrayStore::AxdynU32(to_status_error(list_to_ndarray(
                            slice.to_vec(),
                            shape,
                        ))?)
                    }
                    _ => {
                        return Err(Status::aborted(format!("{inner:?} not supported")));
                    }
                };
                res
            }
            _ => {
                return Err(Status::failed_precondition(
                    "Only series of List type can be converted into a list",
                ));
            }
        };

        Ok(arraystore)
    }
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_tensor(
        &self,
        request: Request<RemoteArray>,
    ) -> Result<Response<Reference>, Status> {
        let identifier = &request.get_ref().identifier;

        let df = self.polars.get_array(&identifier)?;

        let tensor = self.ndarray_to_tensor(df)?;

        let (_, tensor_ref) = self.torch.insert_tensor(Arc::new(Mutex::new(tensor)));

        let tensor_ref = Reference {
            identifier: tensor_ref.identifier,
            meta: tensor_ref.meta,
            ..Default::default()
        };
        Ok(Response::new(tensor_ref))
    }

    async fn conv_to_array(
        &self,
        request: Request<RemoteDataFrame>,
    ) -> Result<Response<RemoteArray>, Status> {
        let identifier = request.into_inner().identifier;

        /*
            Convert to array would have to branch (unless there's a state machine) introduce similar to CompositePlan
            If there aren't strings and lists in the dataframe, and all the types are the same,
            then we use `dataframe_to_ndarray`
        */

        let df = self.polars.get_df_unchecked(&identifier)?;

        let dtypes = df
            .dtypes()
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>();

        let dtype_exists =
            |dtype: &str| -> bool { dtypes.iter().find(|s| s.contains(dtype)).is_some() };
        let arr = if !(dtype_exists("list") || dtype_exists("utf8")) {
            RemoteArray {
                identifier: (self.df_to_ndarray(&df)?),
            }
        } else if dtype_exists("list") {
            /*
               Here, we assume we have a List[PrimitiveType]
               The idea would be to convert columns into ArrayBase -> merge Vec<ArrayBase> -> ArrayBase
            */
            let col_names = df.get_column_names();
            let vec_series = df
                .columns(&col_names[..])
                .map_err(|e| Status::aborted(format!("Could not get Series in DataFrame: {e}")))?;

            let mut out = vec![];
            for series in vec_series {
                out.push(to_status_error(self.list_series_to_array_store(series))?);
            }

            /*
                Here, we stack on Axis(1) because we would want to create [n_rows, m_cols, k_elems_in_each_item];
            */
            let array = ArrayStore::stack(Axis(1), &out[..])?;
            RemoteArray {
                identifier: self.polars.insert_array(array),
            }
        } else {
            return Err(Status::aborted(format!(
                "DataFrame with {dtypes:?} cannot be converted directly to RemoteArray"
            )));
        };

        // In order to not waste memory, we delete the dataframe after the eager `to_array` conversion
        self.polars.delete_dfs(&identifier)?;

        Ok(Response::new(arr))
    }

    async fn tokenize_data_frame(
        &self,
        request: Request<ToTokenizedArrays>,
    ) -> Result<Response<RemoteArrays>, Status> {
        let (identifier, add_special_tokens, model, config, revision, auth_token) = (
            request.get_ref().identifier.clone(),
            request.get_ref().add_special_tokens,
            request.get_ref().model.clone(),
            request.get_ref().config.clone(),
            request.get_ref().revision.clone(),
            request.get_ref().auth_token(),
        );

        let add_special_tokens = add_special_tokens != 0;
        let df = self.polars.get_df_unchecked(&identifier)?;

        let mut identifiers = vec![];
        for name in df.get_column_names_owned() {
            let idx = df
                .get_column_names()
                .iter()
                .position(|x| x == &&name)
                .ok_or(Status::invalid_argument(format!(
                    "Could not apply udf: no column `{}` in data frame",
                    name
                )))?;
            let series = df.get_columns().get(idx).unwrap();

            let (ids, masks) = if series.dtype().eq(&DataType::Utf8) {
                series_to_tokenized_arrays(
                    series,
                    &model,
                    &config,
                    add_special_tokens,
                    &revision,
                    auth_token,
                )?
            } else {
                return Err(Status::aborted(format!(
                    "Non-string columns cannot be tokenized"
                )));
            };

            let ids = RemoteArray {
                identifier: self.polars.insert_array(ArrayStore::AxdynI64(ids)),
            };
            let masks = RemoteArray {
                identifier: self.polars.insert_array(ArrayStore::AxdynI64(masks)),
            };

            identifiers.append(&mut vec![ids, masks]);
        }

        // In order to not waste memory, we delete the dataframe after the eager `to_array` conversion
        self.polars.delete_dfs(&identifier)?;

        Ok(Response::new(RemoteArrays { list: identifiers }))
    }
}
