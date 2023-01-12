use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use linfa_proto::{
    linfa_service_server::LinfaService, ModelResponse, PredictionRequest, ReferenceResponse,
    Trainer, TrainingRequest, ValidationRequest,
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
    bastionlab_polars: Arc<BastionLabPolars>,
    models: Arc<RwLock<HashMap<String, Arc<SupportedModels>>>>,
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
        models.insert(identifier.clone(), Arc::new(model));
        identifier
    }

    fn get_model(&self, identifier: &str) -> Result<Arc<SupportedModels>, Status> {
        let models = self.models.read().unwrap();
        let model = models
            .get(identifier)
            .ok_or(Status::not_found("Model not found!"))?;
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
        let (records, target, trainer): (String, String, Option<Trainer>) =
            process_trainer_req(request)?;

        let (records, target) = {
            let records = self.get_df(&records)?;
            let target = self.get_df(&target)?;
            (records, target)
        };
        let trainer = trainer.ok_or(Status::aborted("Invalid Trainer!"))?.clone();

        let trainer = select_trainer(trainer)?;
        let model = to_status_error(send_to_trainer(records.clone(), target.clone(), trainer))?;
        let identifier = self.insert_model(model);
        Ok(Response::new(ModelResponse { identifier }))
    }

    async fn predict(
        &self,
        request: Request<PredictionRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        self.sess_manager.verify_request(&request)?;
        let (model_id, test_set, probability) = {
            let model = &request.get_ref().model;
            let test_set = &request.get_ref().test_set;
            let prob = *(&request.get_ref().probability);
            (model, test_set, prob)
        };

        let model = self.get_model(model_id)?;

        let test_set = self.bastionlab_polars.get_df_unchecked(test_set)?;
        let prediction = to_status_error(predict(model, test_set, probability))?;

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

    async fn cross_validate(
        &self,
        request: Request<ValidationRequest>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let (trainer, records, targets, cv, scoring) = (
            request.get_ref().trainer.clone(),
            &request.get_ref().records,
            &request.get_ref().targets,
            request.get_ref().cv,
            &request.get_ref().scoring,
        );

        let records = self.bastionlab_polars.get_df_unchecked(records)?;
        let targets = self.bastionlab_polars.get_df_unchecked(targets)?;
        let trainer = trainer.ok_or(Status::aborted("Invalid Trainer!"))?;
        let trainer = select_trainer(trainer)?;

        let df = to_status_error(inner_cross_validate(
            trainer,
            records,
            targets,
            &scoring,
            cv as usize,
        ))?;

        let identifier = self.insert_df(
            DataFrameArtifact::new(df, Policy::allow_by_default(), vec![String::default()])
                .with_fetchable(VerificationResult::Safe),
        );
        let header = self.get_header(&identifier)?;
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}
