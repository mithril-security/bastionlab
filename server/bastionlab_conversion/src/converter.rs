use std::sync::{Arc, Mutex};

use bastionlab_common::common_conversions::*;
use bastionlab_common::session::SessionManager;
use bastionlab_polars::{ArrayStore, BastionLabPolars};
use bastionlab_torch::BastionLabTorch;
use ndarray::{Dim, IxDynImpl, OwnedRepr};
use polars::export::ahash::HashSet;
use polars::prelude::*;
use tch::Tensor;
use tonic::{Request, Response, Status};

use crate::conversion_proto::{conversion_service_server::ConversionService, ToArray};
use crate::conversion_proto::{RemoteArray, RemoteArrays, RemoteDataFrame};

use crate::bastionlab::Reference;
pub struct Converter {
    torch: Arc<BastionLabTorch>,
    polars: Arc<BastionLabPolars>,
    sess_manager: Arc<SessionManager>,
}

impl Converter {
    pub fn new(
        torch: Arc<BastionLabTorch>,
        polars: Arc<BastionLabPolars>,
        sess_manager: Arc<SessionManager>,
    ) -> Self {
        Self {
            torch,
            polars,
            sess_manager,
        }
    }
    pub fn ndarray_to_tensor(&self, arr: ArrayStore) -> Result<Tensor, Status> {
        let tensor = match arr {
            ArrayStore::AxdynI64(a) => ndarray_to_tensor::<OwnedRepr<i64>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynF64(a) => ndarray_to_tensor::<OwnedRepr<f64>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynF32(a) => ndarray_to_tensor::<OwnedRepr<f32>, Dim<IxDynImpl>>(a),
            ArrayStore::AxdynI32(a) => ndarray_to_tensor::<OwnedRepr<i32>, Dim<IxDynImpl>>(a),
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
            _ => {
                return Err(Status::aborted(format!("{:?} not support ", dtype)));
            }
        };
        let identifier = self.polars.insert_array(arr);
        Ok(identifier)
    }
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_tensor(
        &self,
        request: Request<RemoteArray>,
    ) -> Result<Response<Reference>, Status> {
        self.sess_manager.verify_request(&request)?;
        let identifier = &request.into_inner().identifier;

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

        let dtypes = df.dtypes();

        let arr = if !dtypes.contains(&DataType::List(Box::default()))
            && !dtypes.contains(&DataType::Utf8)
        {
            RemoteArray {
                identifier: (self.df_to_ndarray(&df)?),
            }
        } else if dtypes.contains(&DataType::List(Box::default())) {
            /*
               Here, we assume we have a List[PrimitiveType]
            */
            return Err(Status::unavailable(
                "List[Primitive] to ndarray not supported yet",
            ));
        } else {
            return Err(
                Status::aborted("DataFrame with str columns cannot be converted directly to RemoteArray. Please tokenize strings first"));
        };

        Ok(Response::new(arr))
    }

    async fn tokenize_data_frame(
        &self,
        request: Request<ToArray>,
    ) -> Result<Response<RemoteArrays>, Status> {
        let (identifier, add_special_tokens, model, config) = (
            request.get_ref().identifier.clone(),
            request.get_ref().add_special_tokens,
            request.get_ref().model.clone(),
            request.get_ref().config.clone(),
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
                series_to_tokenized_series(series, &model, &config, add_special_tokens)?
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

        Ok(Response::new(RemoteArrays { list: identifiers }))
    }
}
