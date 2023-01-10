use std::sync::Arc;

use bastionlab_common::session::SessionManager;
use bastionlab_common::utils::{kind_to_datatype, tensor_to_series};
use bastionlab_polars::access_control::{Policy, VerificationResult};
use bastionlab_polars::polars_proto::Meta;
use bastionlab_polars::utils::to_status_error;
use bastionlab_polars::utils::{df_to_tensor, series_to_tensor};
use bastionlab_polars::{BastionLabPolars, DataFrameArtifact};
use bastionlab_torch::BastionLabTorch;
use polars::prelude::{DataFrame, DataType};
use prost::Message;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::conversion_proto::{
    conversion_service_server::ConversionService, ConvReference, ConvReferenceResponse,
    ToDataFrame, ToTensor,
};

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
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_data_frame(
        &self,
        request: Request<ToDataFrame>,
    ) -> Result<Response<ConvReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;

        #[allow(unused)]
        let (inputs_cols_names, labels_col_name, identifier, inputs_conv_fn) = {
            (
                &request.get_ref().inputs_col_names,
                &request.get_ref().labels_col_name,
                &request.get_ref().identifier,
                &request.get_ref().inputs_conv_fn,
            )
        };
        let identifier = Uuid::parse_str(&identifier)
            .map_err(|_| Status::invalid_argument("Invalid run reference"))?;

        let datasets = self.torch.datasets.read().unwrap();
        let artifact = datasets
            .get(&identifier.to_string())
            .ok_or(Status::not_found("Dataset not found"))?;
        let data = artifact.data.read().unwrap();

        let samples_inputs_locks = data
            .samples_inputs
            .iter()
            .map(|t| t.lock().unwrap())
            .collect::<Vec<_>>();

        let mut samples_inputs_series = Vec::new();

        for (inputs, name) in samples_inputs_locks.iter().zip(inputs_cols_names.iter()) {
            let dtype = kind_to_datatype(inputs.kind());

            let series = if inputs.size().len() == 2 {
                tensor_to_series(&name, &DataType::List(Box::new(dtype)), inputs.data())?
            } else {
                tensor_to_series(name, &dtype, inputs.data())?
            };
            samples_inputs_series.push(series);
        }

        let labels_lock = data.labels.lock().unwrap();
        let dtype = kind_to_datatype(labels_lock.kind());
        let labels = tensor_to_series(&labels_col_name, &dtype, labels_lock.data())?;

        samples_inputs_series.push(labels);

        let df = to_status_error(DataFrame::new(samples_inputs_series))?;

        let df_artifact = DataFrameArtifact::new(df, Policy::allow_by_default(), vec![]);

        // TODO: Remove this and solve Policy inheritance problem for converted data.
        let df_artifact = df_artifact.with_fetchable(VerificationResult::Safe);

        let identifier = self.polars.insert_df(df_artifact);
        let header = self.polars.get_header(&identifier)?;
        Ok(Response::new(ConvReferenceResponse { identifier, header }))
    }

    async fn conv_to_tensor(
        &self,
        request: Request<ToTensor>,
    ) -> Result<Response<ConvReference>, Status> {
        self.sess_manager.verify_request(&request)?;
        let identifier = &request.get_ref().identifier;

        let df = self.polars.get_df_unchecked(&identifier)?;

        let tensor = {
            let cols = df.get_column_names();
            if cols.len() == 1 {
                let series = to_status_error(df.column(cols[0]))?;
                let data = series_to_tensor(series)?;

                data
            } else {
                let tensor = df_to_tensor(&df)?;
                tensor
            }
        };
        let meta = Meta {
            input_dtype: vec![format!("{:?}", tensor.kind())],
            input_shape: tensor.size(),
            ..Default::default()
        };

        let tensor_id = self.torch.insert_tensor(tensor);

        // Here, a dataframe has been converted and can be cleared from the Polars storage.
        self.polars.delete_dfs(identifier)?;

        Ok(Response::new(ConvReference {
            identifier: tensor_id,
            name: String::new(),
            description: String::new(),
            meta: meta.encode_to_vec(),
        }))
    }
}
