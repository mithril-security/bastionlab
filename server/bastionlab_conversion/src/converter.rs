use std::sync::{Arc, Mutex, RwLock};

use bastionlab_common::utils::{kind_to_datatype, tensor_to_series};
use bastionlab_common::{session::SessionManager, session_proto::ClientInfo};
use bastionlab_learning::data::Dataset;
use bastionlab_polars::access_control::{Policy, VerificationResult};
use bastionlab_polars::polars_proto::Meta;
use bastionlab_polars::utils::{series_to_tensor, vec_series_to_tensor};
use bastionlab_polars::{polars_proto::Reference, utils::to_status_error};
use bastionlab_polars::{BastionLabPolars, DataFrameArtifact};
use bastionlab_torch::storage::Artifact;
use bastionlab_torch::BastionLabTorch;
use polars::prelude::row::{AnyValueBuffer, Row};
use polars::prelude::{AnyValue, DataFrame, DataType};
use polars::series::Series;
use prost::Message;
use ring::hmac;
use tokenizers::{Encoding, PaddingParams, Tokenizer, TokenizerBuilder, TruncationParams};
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

    fn get_tokenizer(&self, model: &str) -> Result<Tokenizer, Status> {
        let tokenizer =
            Tokenizer::from_pretrained(model, None).map_err(|e| Status::aborted(e.to_string()))?;

        let mut t = TokenizerBuilder::new();

        t = t.with_model(tokenizer.get_model().clone());
        t = t.with_normalizer(tokenizer.get_normalizer().cloned());
        t = t.with_pre_tokenizer(tokenizer.get_pre_tokenizer().cloned());
        t = t.with_post_processor(tokenizer.get_post_processor().cloned());
        t = t.with_decoder(tokenizer.get_decoder().cloned());
        t = t.with_truncation(
            tokenizer
                .get_truncation()
                .cloned()
                .or_else(|| Some(TruncationParams::default())),
        );
        t = t.with_padding(
            tokenizer
                .get_padding()
                .cloned()
                .or_else(|| Some(PaddingParams::default())),
        );
        let tokenizer = t.build().map_err(|e| Status::aborted(e.to_string()))?;
        let tokenizer: Tokenizer = tokenizer.into();
        Ok(tokenizer)
    }

    pub fn series_to_tokenized_series(
        &self,
        s: &Series,
        name: &str,
        model: &str,
    ) -> Result<DataFrame, Status> {
        let tokenizer = self.get_tokenizer(model)?;
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

    pub fn insert_torch_dataset(
        &self,
        df: &DataFrame,
        inputs: &Vec<String>,
        labels: &str,
        client_info: Option<ClientInfo>,
        model: &str,
    ) -> Result<Reference, Status> {
        let inputs = to_status_error(df.columns(&inputs[..]))?;
        let labels = to_status_error(df.column(labels))?;

        // Remove all Series<Utf8Type> into another vec
        let not_utf8_inputs = inputs
            .iter()
            .filter_map(|s| {
                if s.dtype().ne(&DataType::Utf8) {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut utf8_inputs = inputs
            .iter()
            .filter_map(|s| {
                if s.dtype().eq(&DataType::Utf8) {
                    Some((s.clone(), s.name()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut out = vec![];
        let mut out_shapes = vec![];
        let mut out_dtypes = vec![];

        for (s, n) in utf8_inputs.drain(..) {
            let df = self.series_to_tokenized_series(s, n, model)?;
            let cols = df.get_column_names();
            let vs = to_status_error(df.columns(&cols[..]))?;
            let (mut vs, mut vs_shapes, mut vs_dtypes, _) = vec_series_to_tensor(vs)?;
            out.append(&mut vs);
            out_shapes.append(&mut vs_shapes);
            out_dtypes.append(&mut vs_dtypes);
        }

        let (mut inputs, mut shapes, mut dtypes, nb_samples) =
            vec_series_to_tensor(not_utf8_inputs)?;

        let labels = Mutex::new(series_to_tensor(labels)?);
        let identifier = format!("{}", Uuid::new_v4());

        inputs.append(&mut out);
        shapes.append(&mut out_shapes);
        dtypes.append(&mut out_dtypes);

        let data = Dataset::new(inputs, labels);
        let meta = Meta {
            input_shape: shapes,
            input_dtype: dtypes,
            nb_samples: nb_samples.try_into().unwrap(),
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
        Ok(Reference {
            identifier,
            name,
            description,
            meta,
        })
    }
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_data_frame(
        &self,
        request: Request<ToDataFrame>,
    ) -> Result<Response<ConvReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;

        let (inputs_cols_names, labels_col_name, identifier) = {
            (
                &request.get_ref().inputs_col_names,
                &request.get_ref().labels_col_name,
                &request.get_ref().identifier,
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
            samples_inputs_series.push(tensor_to_series(&name, &dtype, inputs.data())?);
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
        let (inputs, labels, identifier, inputs_conv_fn) = {
            (
                &request.get_ref().inputs,
                &request.get_ref().labels,
                &request.get_ref().identifier,
                &request.get_ref().inputs_conv_fn,
            )
        };

        let df = self.polars.get_df_unchecked(&identifier)?;
        let Reference {
            identifier,
            name,
            description,
            meta,
        } = self.insert_torch_dataset(&df, inputs, labels, Some(client_info), &inputs_conv_fn)?;

        Ok(Response::new(ConvReference {
            identifier,
            name,
            description,
            meta,
        }))
    }
}
