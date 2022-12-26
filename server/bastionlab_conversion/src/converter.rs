use std::sync::{Arc, Mutex, RwLock};

use bastionlab_common::utils::{kind_to_datatype, tensor_to_series};
use bastionlab_common::{session::SessionManager, session_proto::ClientInfo};
use bastionlab_learning::data::Dataset;
use bastionlab_polars::access_control::{Policy, VerificationResult};
use bastionlab_polars::polars_proto::Meta;
use bastionlab_polars::utils::{
    series_to_tensor, tokenized_series_to_series, vec_series_to_tensor,
};
use bastionlab_polars::{polars_proto::Reference, utils::to_status_error};
use bastionlab_polars::{BastionLabPolars, DataFrameArtifact};
use bastionlab_torch::storage::Artifact;
use bastionlab_torch::BastionLabTorch;
use polars::prelude::{DataFrame, DataType};
use prost::Message;
use ring::hmac;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::conversion_proto::{
    conversion_service_server::ConversionService, ConvReference, ConvReferenceResponse,
    ToDataFrame, ToDataset,
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

    pub fn insert_torch_dataset(
        &self,
        df: &DataFrame,
        inputs: &Vec<String>,
        labels: &str,
        client_info: Option<ClientInfo>,
    ) -> Result<Reference, Status> {
        let inputs = to_status_error(df.columns(&inputs[..]))?;
        let labels = to_status_error(df.column(labels))?;

        let (inputs, shapes, dtypes, nb_samples) = vec_series_to_tensor(inputs)?;

        let labels = Mutex::new(series_to_tensor(labels)?.data());
        let identifier = format!("{}", Uuid::new_v4());

        let data = Dataset::new(inputs, labels, true, (vec![], "".to_string()));
        let meta = Meta {
            input_shape: shapes,
            input_dtype: dtypes,
            nb_samples,
            privacy_limit: 0.0,
            train_dataset: None,
        };

        let dataset = Artifact {
            data: Arc::new(RwLock::new(data)),
            name: String::default(),
            description: String::default(),
            secret: hmac::Key::new(ring::hmac::HMAC_SHA256, &[0]),
            meta: meta.encode_to_vec(),
            client_info,
        };
        let (identifier, name, description, meta) =
            self.torch.insert_dataset(&identifier, dataset)?;

        let train_dataset = Reference {
            identifier: identifier.clone(),
            name,
            description,
            meta,
        };

        Ok(train_dataset)
    }
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_data_frame(
        &self,
        request: Request<ToDataFrame>,
    ) -> Result<Response<ConvReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;

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
            println!("Dtype: {}", dtype);

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

    async fn conv_to_dataset(
        &self,
        request: Request<ToDataset>,
    ) -> Result<Response<ConvReference>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let (inputs, labels, identifier) = {
            (
                &request.get_ref().inputs,
                &request.get_ref().labels,
                &request.get_ref().identifier,
            )
        };

        let df = self.polars.get_df_unchecked(&identifier)?;
        let Reference {
            identifier,
            name,
            description,
            meta,
        } = self.insert_torch_dataset(&df, inputs, labels, Some(client_info))?;

        Ok(Response::new(ConvReference {
            identifier,
            name,
            description,
            meta,
        }))
    }
}
