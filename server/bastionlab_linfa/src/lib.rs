use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use linfa_proto::{
    linfa_service_server::LinfaService, training_request::Trainer, ModelResponse,
    PredictionRequest, ReferenceResponse, TrainingRequest,
};
pub mod linfa_proto {
    tonic::include_proto!("bastionlab_linfa");
}

use polars::prelude::DataFrame;

mod trainer;
use trainer::{process_trainer_req, select_trainer, SupportedModels};

mod algorithms;

mod operations;
use operations::*;
use uuid::Uuid;

use std::error::Error;

use tonic::{Request, Response, Status};

use bastionlab_common::session::SessionManager;
use bastionlab_polars::{
    access_control::{Policy, VerificationResult},
    BastionLabPolars, DataFrameArtifact,
};

pub fn to_status_error<T, E: Error>(input: Result<T, E>) -> Result<T, Status> {
    input.map_err(|err| Status::aborted(err.to_string()))
}

pub struct BastionLabLinfa {
    bastionlab_polars: Arc<BastionLabPolars>, // Fix by replacing with RdLock
    models: Arc<RwLock<HashMap<String, SupportedModels>>>,
    sess_manager: Arc<SessionManager>,
}

impl BastionLabLinfa {
    pub fn new(sess_manager: Arc<SessionManager>, bl_polars: BastionLabPolars) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            sess_manager,
            bastionlab_polars: Arc::new(bl_polars),
        }
    }
    fn get_df(&self, identifier: &str) -> Result<DataFrame, Status> {
        self.bastionlab_polars.get_df_unchecked(identifier)
    }

    pub fn insert_df(&self, df: DataFrameArtifact) -> String {
        self.bastionlab_polars.insert_df(df)
    }
    fn get_header(&self, identifier: &str) -> Result<String, Status> {
        self.bastionlab_polars.get_header(identifier)
    }

    fn insert_model(&self, model: SupportedModels) -> String {
        let mut models = self.models.write().unwrap();
        let identifier = format!("{}", Uuid::new_v4());
        models.insert(identifier.clone(), model);
        identifier
    }
    fn get_model(&self, identifier: &str) -> Result<SupportedModels, Status> {
        let models = self.models.read().unwrap();
        let model = models.get(identifier).unwrap();
        Ok(model.clone())
    }
}

#[tonic::async_trait]
impl LinfaService for BastionLabLinfa {
    async fn train(
        &self,
        request: Request<TrainingRequest>,
    ) -> Result<Response<ModelResponse>, Status> {
        self.sess_manager.verify_request(&request)?;
        let (records, target, ratio, trainer): (String, String, f32, Option<Trainer>) =
            process_trainer_req(request)?;

        let (records, target) = {
            let records = self.get_df(&records)?;
            let target = self.get_df(&target)?;
            (records, target)
        };

        let trainer = trainer.ok_or(Status::aborted("Invalid Trainer!"))?;
        let trainer = select_trainer(trainer)?;
        let model = to_status_error(send_to_trainer(
            records.clone(),
            target.clone(),
            ratio,
            trainer,
        ))?;
        let identifier = self.insert_model(model);
        Ok(Response::new(ModelResponse { identifier }))
    }

    async fn predict(
        &self,
        request: Request<PredictionRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;
        let (model_id, data) = {
            let model = &request.get_ref().model;
            let data = &request.get_ref().data;
            let data = to_type! {<f64>(data)};
            (model, data)
        };

        let model = self.get_model(model_id)?;

        let prediction = to_status_error(predict(model, data))?;

        println!("{:?}", prediction);

        let identifier = self.insert_df(
            DataFrameArtifact::new(
                prediction,
                Policy::allow_by_default(),
                vec![String::default()],
            )
            .with_fetchable(VerificationResult::Safe),
        );
        let header = self.get_header(&identifier)?;
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}
