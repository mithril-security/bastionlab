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

mod trainers;
use trainers::{select_trainer, SupportedModels};

mod algorithms;

mod operations;
use operations::*;

mod utils;
use utils::process_trainer_req;

use uuid::Uuid;

use std::error::Error;

use tonic::{Request, Response, Status};

use bastionlab_polars::{
    access_control::{Policy, VerificationResult},
    BastionLabPolars, DataFrameArtifact,
};

pub fn to_status_error<T, E: Error>(input: Result<T, E>) -> Result<T, Status> {
    input.map_err(|err| Status::aborted(err.to_string()))
}

pub struct BastionLabLinfa {
    polars: Arc<BastionLabPolars>,
    models: Arc<RwLock<HashMap<String, Arc<SupportedModels>>>>,
}

impl BastionLabLinfa {
    pub fn new(polars: BastionLabPolars) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            polars: Arc::new(polars),
        }
    }

    pub fn insert_df(&self, df: DataFrameArtifact) -> String {
        self.polars.insert_df(df)
    }
    fn get_header(&self, identifier: &str) -> Result<String, Status> {
        self.polars.get_header(identifier)
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
        let (records, target, trainer): (String, String, Option<Trainer>) =
            process_trainer_req(request)?;

        let (records, target) = {
            let records = self.polars.get_array(&records)?;
            let target = self.polars.get_array(&target)?;
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
        let (model_id, input, probability) = {
            let model = &request.get_ref().model;
            let input = &request.get_ref().input;
            let prob = *(&request.get_ref().probability);
            (model, input, prob)
        };

        let model = self.get_model(model_id)?;

        let input = self.polars.get_array(input)?;
        let prediction = to_status_error(predict(model, input, probability))?;

        let prediction_titles = |width: usize| {
            let mut titles = vec![];

            for i in 0..width {
                titles.push(format!("Class{i}"));
            }
            return titles;
        };

        let prediction = prediction.to_dataframe(prediction_titles(prediction.width()))?;

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

        let records = self.polars.get_array(records)?;
        let targets = self.polars.get_array(targets)?;
        let trainer = trainer.ok_or(Status::aborted("Invalid Trainer!"))?;
        let trainer = select_trainer(trainer)?;

        let df = to_status_error(inner_cross_validate(
            trainer,
            records,
            targets,
            &scoring,
            cv as usize,
        ))?
        .to_dataframe(vec![scoring.to_string()])?;

        // FIXME: Currently, everyone can fetch predictions.
        // Ideally, we would want policies to trickly down here too
        // TODO: Figure out how to pass policies to the `ndarray` -> `DataFrame` problem
        //       where there wasn't any policy to inherit from in the first place.
        //       Possible solution:
        //         - Add a `policy` field to `ArrayStore` and this will inherit the policy of the
        //           DataFrame, from which it was converted.
        let identifier = self.insert_df(
            DataFrameArtifact::new(df, Policy::allow_by_default(), vec![String::default()])
                .with_fetchable(VerificationResult::Safe),
        );
        let header = self.get_header(&identifier)?;
        Ok(Response::new(ReferenceResponse { identifier, header }))
    }
}
