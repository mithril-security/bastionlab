use bastionlab_common::prelude::*;
use bastionlab_common::session::get_token;
use bastionlab_common::session::SessionManager;
use bastionlab_common::telemetry::{self, TelemetryEventProps};
use bastionlab_learning::nn::Module;
use bastionlab_learning::{data::Dataset, nn::CheckPoint};
use prost::Message;
use ring::digest;
use ring::hmac;
use std::time::Instant;
use tch::Tensor;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod torch_proto {
    tonic::include_proto!("bastionlab_torch");
}
use torch_proto::torch_service_server::TorchService;
use torch_proto::{
    Chunk, Devices, Empty, Meta, Metric, Optimizers, Reference, References, TestConfig,
    TrainConfig, UpdateMeta,
};

pub mod storage;
use storage::Artifact;

mod utils;
use utils::*;

mod learning;
use learning::*;

mod serialization;
use serialization::*;

use bastionlab_learning::serialization::{BinaryModule, SizedObjectsBytes};

/// The server's state
#[derive(Clone)]
pub struct BastionLabTorch {
    binaries: Arc<RwLock<HashMap<String, Artifact<BinaryModule>>>>,
    checkpoints: Arc<RwLock<HashMap<String, Artifact<CheckPoint>>>>,
    pub datasets: Arc<RwLock<HashMap<String, Artifact<Dataset>>>>,
    runs: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Run>>>>>,
    sess_manager: Arc<SessionManager>,
    tensors: Arc<RwLock<HashMap<String, Mutex<Tensor>>>>,
}

impl BastionLabTorch {
    pub fn new(sess_manager: Arc<SessionManager>) -> Self {
        BastionLabTorch {
            binaries: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            runs: Arc::new(RwLock::new(HashMap::new())),
            tensors: Arc::new(RwLock::new(HashMap::new())),
            sess_manager,
        }
    }

    pub fn insert_dataset(
        &self,
        identifier: &str,
        dataset: Artifact<Dataset>,
    ) -> Result<(String, String, String, Vec<u8>), Status> {
        let identifier = identifier.to_string();
        let name = dataset.name.clone();
        let description = dataset.description.clone();
        let meta = dataset.meta.clone();

        self.datasets
            .write()
            .unwrap()
            .insert(identifier.clone(), dataset);
        info!(
            "Successfully inserted dataset {} in server",
            identifier.clone()
        );

        Ok((identifier.clone(), name, description, meta))
    }

    pub fn insert_tensor(&self, tensor: Tensor) -> String {
        let identifier = Uuid::new_v4();
        self.tensors
            .write()
            .unwrap()
            .insert(identifier.to_string(), Mutex::new(tensor));

        info!("Successfully inserted tensor {}", identifier);
        identifier.to_string()
    }

    fn get_tensor(&self, identifier: &str) -> Result<Tensor, Status> {
        let tensors = self.tensors.read().unwrap();
        let tensor = tensors
            .get(identifier)
            .ok_or(Status::aborted("Could not find tensor on BastionLab Torch"))?;

        let tensor = tensor.lock().unwrap();

        let tensor = { tensor.data() };

        Ok(tensor)
    }

    pub fn update_dataset(
        &self,
        identifier: &str,
        name: Option<String>,
        meta: Option<Meta>,
        description: Option<String>,
    ) -> Result<(), Status> {
        let mut datasets = self.datasets.write().unwrap();

        let dataset = datasets.get_mut(identifier);
        match dataset {
            Some(v) => {
                if let Some(name) = name {
                    v.name = name;
                }
                if let Some(meta) = meta {
                    v.meta = meta.encode_to_vec();
                }
                if let Some(description) = description {
                    v.description = description;
                }
                Ok(())
            }
            None => {
                return Err(Status::aborted("Dataset not found!"));
            }
        }
    }

    fn convert_from_remote_dataset(
        &self,
        serialized_dataset: &str,
    ) -> Result<(Arc<RwLock<Dataset>>, String), Status> {
        let dataset: RemoteDataset = serde_json::from_str(&serialized_dataset).map_err(|e| {
            Status::invalid_argument(format!("Could not deserialize RemoteDataset: {}", e))
        })?;

        let hash =
            hex::encode(digest::digest(&digest::SHA256, serialized_dataset.as_bytes()).as_ref());

        let mut samples_inputs = vec![];

        for input in dataset.inputs {
            samples_inputs.push(Mutex::new(self.get_tensor(&input.identifier)?));
        }

        let label = Mutex::new(self.get_tensor(&dataset.label.identifier)?);
        let limit = dataset.privacy_limit;
        
        let data = Dataset::new(samples_inputs, label, limit);

        Ok((Arc::new(RwLock::new(data)), hash))
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
        let token = get_token(&request, self.sess_manager.auth_enabled())?;
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
        let (_, name, description, meta) = self.insert_dataset(&dataset_hash, dataset)?;

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

        let token = get_token(&request, self.sess_manager.auth_enabled())?;
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
                .ok_or(Status::not_found("Dataset not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 4_194_285, "Dataset".to_string()).await)
    }

    async fn fetch_module(
        &self,
        request: Request<Reference>,
    ) -> Result<Response<Self::FetchModuleStream>, Status> {
        let token = get_token(&request, self.sess_manager.auth_enabled())?;
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
                        .ok_or(Status::not_found("Module not found!"))?;
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
        let token = get_token(&request, self.sess_manager.auth_enabled())?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let config = request.into_inner();
        // let d = RemoteDataset {
        //     inputs: vec![RemoteTensor {
        //         identifier: "".to_string(),
        //     }],
        //     label: RemoteTensor {
        //         identifier: "".to_string(),
        //     },
        //     nb_samples: 1,
        //     privacy_limit: -1.0,
        // };

        let (dataset, dataset_id) = self.convert_from_remote_dataset(&config.dataset)?;

        let binary_id = config
            .model
            .clone()
            .ok_or(Status::invalid_argument("Invalid module reference"))?
            .identifier;
        let device = parse_device(&config.device)?;

        let (binary, chkpt) = {
            let binaries = self.binaries.read().unwrap();
            let binary: &Artifact<BinaryModule> = binaries
                .get(&binary_id)
                .ok_or(Status::not_found("Module binary not found"))?;
            let mut checkpoints = self.checkpoints.write().unwrap();
            let chkpt = if config.resume {
                let chkpt = checkpoints
                    .get(&binary_id)
                    .ok_or(Status::not_found("CheckPoint not found!"))?;
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
                    .ok_or(Status::not_found("Module binary not found"))?;
                chkpt
            };
            (Arc::clone(&binary.data), Arc::clone(&chkpt.data))
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
        let token = get_token(&request, self.sess_manager.auth_enabled())?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let config = request.into_inner();
        let dataset_id = config
            .dataset
            .clone()
            .ok_or(Status::invalid_argument("Invalid dataset reference"))?
            .identifier;
        let module_id = config
            .model
            .clone()
            .ok_or(Status::invalid_argument("Invalid dataset reference"))?
            .identifier;
        let device = parse_device(&config.device)?;
        let (module, binary) = {
            let chkpts_store = self.checkpoints.read().unwrap();
            let artifact = chkpts_store
                .get(&module_id)
                .ok_or(Status::not_found("Module not found"))?;
            let binaries = self.binaries.read().unwrap();
            let binary = binaries.get(&module_id).unwrap();

            (Arc::clone(&artifact.data), Arc::clone(&binary.data))
        };

        let dataset = {
            let datasets = self.datasets.read().unwrap();
            let dataset = datasets
                .get(&dataset_id)
                .ok_or(Status::not_found("Dataset not found"))?;
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

    async fn update_dataset(
        &self,
        request: Request<UpdateMeta>,
    ) -> Result<Response<Empty>, Status> {
        let (identifier, name, description, meta) = (
            request.get_ref().identifier.clone(),
            request.get_ref().name.clone(),
            request.get_ref().description.clone(),
            request.get_ref().meta.clone(),
        );

        self.update_dataset(&identifier, name, meta, description)?;
        Ok(Response::new(Empty {}))
    }

    async fn send_tensor(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<Reference>, Status> {
        let res = unstream_data(request.into_inner()).await?;

        let (tensor, meta) = {
            let data = res.data.read().unwrap();
            let data: Tensor = (&*data).try_into().unwrap();
            println!("{:?}", data.size());
            let meta = Meta {
                input_dtype: vec![format!("{:?}", data.kind())],
                input_shape: data.size(),
                ..Default::default()
            };
            (data, meta)
        };

        let identifier = self.insert_tensor(tensor);
        Ok(Response::new(Reference {
            identifier,
            name: String::new(),
            description: String::new(),
            meta: meta.encode_to_vec(),
        }))
    }
}
