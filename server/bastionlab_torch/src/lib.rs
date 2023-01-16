use bastionlab_common::prelude::*;
use bastionlab_common::session::SessionManager;
use bastionlab_common::telemetry::{self, TelemetryEventProps};
use bastionlab_learning::nn::Module;
use bastionlab_learning::{data::Dataset, nn::CheckPoint};
use ring::digest;
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod torch_proto {
    tonic::include_proto!("bastionlab_torch");
}
use torch_proto::torch_service_server::TorchService;
use torch_proto::{
    Chunk, Devices, Empty, Metric, Optimizers, Reference, References, TestConfig, TrainConfig,
};

mod storage;
use storage::Artifact;

mod utils;
use utils::*;

mod learning;
use learning::*;

mod serialization;
use serialization::*;

use bastionlab_learning::serialization::{BinaryModule, SizedObjectsBytes};

/// The server's state
pub struct BastionLabTorch {
    binaries: RwLock<HashMap<String, Artifact<BinaryModule>>>,
    checkpoints: RwLock<HashMap<String, Artifact<CheckPoint>>>,
    datasets: RwLock<HashMap<String, Artifact<Dataset>>>,
    runs: RwLock<HashMap<Uuid, Arc<RwLock<Run>>>>,
    sess_manager: Arc<SessionManager>,
}

impl BastionLabTorch {
    pub fn new(sess_manager: Arc<SessionManager>) -> Self {
        BastionLabTorch {
            binaries: RwLock::new(HashMap::new()),
            checkpoints: RwLock::new(HashMap::new()),
            datasets: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
            sess_manager,
        }
    }
}

#[tonic::async_trait]
impl TorchService for BastionLabTorch {
    type FetchDatasetStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchModuleStream = ReceiverStream<Result<Chunk, Status>>;

    async fn send_dataset(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<Reference>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let start_time = Instant::now();

        let artifact: Artifact<SizedObjectsBytes> = unstream_data(request.into_inner()).await?;

        let (dataset_hash, dataset_size) = {
            let lock = artifact.data.read().unwrap();
            let data = lock.get();
            let hash = hex::encode(digest::digest(&digest::SHA256, &data).as_ref());
            (hash, data.len())
        };

        let dataset: Artifact<Dataset> = tcherror_to_status((artifact).deserialize())?;
        let name = dataset.name.clone();
        let description = dataset.description.clone();
        let meta = dataset.meta.clone();

        self.datasets
            .write()
            .unwrap()
            .insert(dataset_hash.clone(), dataset);

        let elapsed = start_time.elapsed();
        info!("Upload Dataset successful in {}ms", elapsed.as_millis());

        telemetry::add_event(
            TelemetryEventProps::SendDataset {
                dataset_name: Some(name.clone()),
                dataset_size,
                time_taken: elapsed.as_millis() as f64,
                dataset_hash: Some(dataset_hash.clone()),
            },
            Some(client_info),
        );
        Ok(Response::new(Reference {
            identifier: format!("{}", dataset_hash),
            name,
            description,
            meta,
        }))
    }

    async fn send_model(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<Reference>, Status> {
        let start_time = Instant::now();

        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let artifact: Artifact<SizedObjectsBytes> = unstream_data(request.into_inner()).await?;

        let (model_hash, model_size) = {
            let lock = artifact.data.read().unwrap();
            let data = lock.get();
            let hash = hex::encode(digest::digest(&digest::SHA256, &data).as_ref());
            (hash, data.len())
        };

        let binary = tcherror_to_status(artifact.deserialize())?;

        let name = binary.name.clone();
        let description = binary.description.clone();
        let meta = binary.meta.clone();

        self.binaries
            .write()
            .unwrap()
            .insert(model_hash.clone(), binary);
        let elapsed = start_time.elapsed();

        info!("Upload Model successful in {}ms", elapsed.as_millis());

        telemetry::add_event(
            TelemetryEventProps::SendModel {
                model_name: Some(name.clone()),
                model_hash: Some(model_hash.clone()),
                model_size,
                time_taken: elapsed.as_millis() as f64,
            },
            Some(client_info),
        );
        Ok(Response::new(Reference {
            identifier: format!("{}", model_hash),
            name,
            description,
            meta,
        }))
    }

    async fn fetch_dataset(
        &self,
        request: Request<Reference>,
    ) -> Result<Response<Self::FetchDatasetStream>, Status> {
        let identifier = request.into_inner().identifier;
        let serialized = {
            let datasets = self.datasets.read().unwrap();
            let artifact = datasets
                .get(&identifier)
                .ok_or_else(|| Status::not_found("Dataset not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 4_194_285, "Dataset".to_string()).await)
    }

    async fn fetch_module(
        &self,
        request: Request<Reference>,
    ) -> Result<Response<Self::FetchModuleStream>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let identifier = request.into_inner().identifier;

        let serialized = {
            let checkpoints = self.checkpoints.read().unwrap();

            let checkpoint = checkpoints.get(&identifier);
            match checkpoint {
                Some(chkpt) => {
                    let artifact = chkpt;
                    let checkpoints = &artifact.data.read().unwrap().data;
                    let last_chkpt = &checkpoints[checkpoints.len() - 1];

                    let mut chkpt_bytes = SizedObjectsBytes::new();
                    chkpt_bytes.append_back(last_chkpt.clone());

                    Artifact {
                        data: Arc::new(RwLock::new(chkpt_bytes)),
                        name: artifact.name.clone(),
                        client_info: Some(client_info),
                        secret: artifact.secret.clone(),
                        description: artifact.description.clone(),
                        meta: artifact.meta.clone(),
                    }
                }
                None => {
                    let binaries = self.binaries.read().unwrap();
                    let binary = binaries
                        .get(&identifier)
                        .ok_or_else(|| Status::not_found("Module not found!"))?;
                    let module: Module = (&*binary.data.read().unwrap()).try_into().unwrap();
                    let module = Artifact {
                        data: Arc::new(RwLock::new(module)),
                        name: binary.name.clone(),
                        client_info: Some(client_info),
                        secret: binary.secret.clone(),
                        description: binary.description.clone(),
                        meta: binary.meta.clone(),
                    };
                    tcherror_to_status(module.serialize())?
                }
            }
        };

        Ok(stream_data(serialized, 4_194_285, "Model".to_string()).await)
    }

    async fn delete_dataset(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = request.into_inner().identifier;
        self.datasets.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn delete_module(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = request.into_inner().identifier;
        self.binaries.write().unwrap().remove(&identifier);
        self.checkpoints.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn train(&self, request: Request<TrainConfig>) -> Result<Response<Reference>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let config = request.into_inner();
        let dataset_id = config
            .dataset
            .clone()
            .ok_or_else(|| Status::invalid_argument("Invalid dataset reference"))?
            .identifier;
        let binary_id = config
            .model
            .clone()
            .ok_or_else(|| Status::invalid_argument("Invalid module reference"))?
            .identifier;
        let device = parse_device(&config.device)?;
        let (binary, chkpt) = {
            let binaries = self.binaries.read().unwrap();
            let binary: &Artifact<BinaryModule> = binaries
                .get(&binary_id)
                .ok_or_else(|| Status::not_found("Module binary not found"))?;
            let mut checkpoints = self.checkpoints.write().unwrap();
            let chkpt = if config.resume {
                let chkpt = checkpoints
                    .get(&binary_id)
                    .ok_or_else(|| Status::not_found("CheckPoint not found!"))?;
                chkpt
            } else {
                let chkpt = Artifact {
                    data: Arc::new(RwLock::new(CheckPoint::new(config.eps >= 0.0))),
                    name: binary.name.clone(),
                    client_info: Some(client_info.clone()),
                    secret: binary.secret.clone(),
                    description: binary.description.clone(),
                    meta: binary.meta.clone(),
                };
                checkpoints.insert(binary_id.clone(), chkpt);
                let chkpt = checkpoints
                    .get(&binary_id)
                    .ok_or_else(|| Status::not_found("Module binary not found"))?;
                chkpt
            };
            (Arc::clone(&binary.data), Arc::clone(&chkpt.data))
        };
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            let dataset = datasets
                .get(&dataset_id)
                .ok_or_else(|| Status::not_found("Dataset not found"))?;
            Arc::clone(&dataset.data)
        };

        let identifier = Uuid::new_v4();
        self.runs
            .write()
            .unwrap()
            .insert(identifier, Arc::new(RwLock::new(Run::Pending)));
        let run = Arc::clone(self.runs.read().unwrap().get(&identifier).unwrap());
        module_train(
            binary,
            dataset,
            run,
            config,
            device,
            binary_id,
            dataset_id,
            Some(client_info),
            chkpt,
        );
        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name: format!("Run #{}", identifier),
            description: String::from(""),
            meta: Vec::new(),
        }))
    }

    async fn test(&self, request: Request<TestConfig>) -> Result<Response<Reference>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let config = request.into_inner();
        let dataset_id = config
            .dataset
            .clone()
            .ok_or_else(|| Status::invalid_argument("Invalid dataset reference"))?
            .identifier;
        let module_id = config
            .model
            .clone()
            .ok_or_else(|| Status::invalid_argument("Invalid dataset reference"))?
            .identifier;
        let device = parse_device(&config.device)?;
        let (module, binary) = {
            let chkpts_store = self.checkpoints.read().unwrap();
            let artifact = chkpts_store
                .get(&module_id)
                .ok_or_else(|| Status::not_found("Module not found"))?;
            let binaries = self.binaries.read().unwrap();
            let binary = binaries.get(&module_id).unwrap();

            (Arc::clone(&artifact.data), Arc::clone(&binary.data))
        };

        let dataset = {
            let datasets = self.datasets.read().unwrap();
            let dataset = datasets
                .get(&dataset_id)
                .ok_or_else(|| Status::not_found("Dataset not found"))?;
            Arc::clone(&dataset.data)
        };

        let identifier = Uuid::new_v4();
        self.runs
            .write()
            .unwrap()
            .insert(identifier, Arc::new(RwLock::new(Run::Pending)));
        let run = Arc::clone(self.runs.read().unwrap().get(&identifier).unwrap());
        module_test(
            module,
            binary,
            dataset,
            run,
            config,
            device,
            module_id,
            dataset_id,
            Some(client_info),
        );
        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name: format!("Run #{}", identifier),
            description: String::from(""),
            meta: Vec::new(),
        }))
    }

    async fn available_models(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<References>, Status> {
        let list = self
            .binaries
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| Reference {
                identifier: format!("{}", k),
                name: v.name.clone(),
                description: v.description.clone(),
                meta: v.meta.clone(),
            })
            .collect();

        Ok(Response::new(References { list }))
    }

    async fn available_datasets(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<References>, Status> {
        let list = self
            .datasets
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| Reference {
                identifier: format!("{}", k),
                name: v.name.clone(),
                description: v.description.clone(),
                meta: v.meta.clone(),
            })
            .collect();

        Ok(Response::new(References { list }))
    }

    async fn available_devices(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Devices>, Status> {
        let mut list = vec![String::from("cpu")];
        if tch::Cuda::is_available() {
            list.push(String::from("gpu"));
            for index in 0..tch::Cuda::device_count() {
                list.push(format!("cuda:{}", index));
            }
        }

        Ok(Response::new(Devices { list }))
    }

    async fn available_optimizers(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Optimizers>, Status> {
        let list = vec!["SGD", "Adam"].iter().map(|v| v.to_string()).collect();
        Ok(Response::new(Optimizers { list }))
    }

    async fn get_metric(&self, request: Request<Reference>) -> Result<Response<Metric>, Status> {
        let identifier = Uuid::parse_str(&request.into_inner().identifier)
            .map_err(|_| Status::invalid_argument("Invalid run reference"))?;

        match &*self
            .runs
            .read()
            .unwrap()
            .get(&identifier)
            .unwrap()
            .read()
            .unwrap()
        {
            Run::Pending => Err(Status::out_of_range("Run has not started.")),
            Run::Ok(m) => Ok(Response::new(m.clone())),
            Run::Error(e) => Err(Status::internal(e.message())),
        }
    }
}
