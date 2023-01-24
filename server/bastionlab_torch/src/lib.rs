use bastionlab_common::prelude::*;
use bastionlab_common::session::SessionManager;
use bastionlab_common::telemetry::{self, TelemetryEventProps};
use bastionlab_learning::nn::Module;
use bastionlab_learning::{data::Dataset, data::DatasetMetadata, nn::CheckPoint};
use prost::Message;
use ring::digest;
use ring::hmac::Key;
use std::time::Instant;
use tch::Tensor;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod torch_proto {
    tonic::include_proto!("bastionlab_torch");
}

pub mod bastionlab {
    tonic::include_proto!("bastionlab");
}

use torch_proto::torch_service_server::TorchService;
use torch_proto::{
    Chunk, Devices, Empty, Metric, Optimizers, References, RemoteDatasetReference, TestConfig,
    TrainConfig, UpdateTensor,
};

use bastionlab::{Reference, TensorMetaData};
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
    datasets_metadata: Arc<RwLock<HashMap<String, DatasetMetadata>>>,
    dataset_remote_dataset_relation: Arc<RwLock<HashMap<String, RemoteDatasetReference>>>,
    runs: Arc<RwLock<HashMap<Uuid, Arc<RwLock<Run>>>>>,
    sess_manager: Arc<SessionManager>,
    tensors: Arc<RwLock<HashMap<String, Arc<Mutex<Tensor>>>>>,
}

impl BastionLabTorch {
    pub fn new(sess_manager: Arc<SessionManager>) -> Self {
        BastionLabTorch {
            binaries: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            datasets_metadata: Arc::new(RwLock::new(HashMap::new())),
            runs: Arc::new(RwLock::new(HashMap::new())),
            tensors: Arc::new(RwLock::new(HashMap::new())),
            dataset_remote_dataset_relation: Arc::new(RwLock::new(HashMap::new())),
            sess_manager,
        }
    }

    pub fn insert_dataset_metadata(&self, artifact: &Artifact<Dataset>) -> DatasetMetadata {
        let mut metadata = self.datasets_metadata.write().unwrap();

        let identifier = Uuid::new_v4().to_string();
        let dataset_metadata = DatasetMetadata {
            identifier: identifier.clone(),
            description: artifact.description.clone(),
            name: artifact.name.clone(),
            meta: artifact.meta.clone(),
        };

        metadata.insert(identifier, dataset_metadata.clone());

        dataset_metadata
    }

    pub fn insert_tensor(&self, tensor: Tensor) -> String {
        let identifier = Uuid::new_v4();
        self.tensors
            .write()
            .unwrap()
            .insert(identifier.to_string(), Arc::new(Mutex::new(tensor)));

        info!("Successfully inserted tensor {}", identifier);
        identifier.to_string()
    }

    fn get_tensor(&self, identifier: &str) -> Result<Arc<Mutex<Tensor>>, Status> {
        let tensors = self.tensors.read().unwrap();
        let tensor = tensors
            .get(identifier)
            .ok_or(Status::aborted("Could not find tensor on BastionLab Torch"))?;

        Ok(Arc::clone(tensor))
    }

    fn convert_from_remote_dataset(
        &self,
        serialized_dataset: &str,
    ) -> Result<(Arc<RwLock<Dataset>>, String, Option<DatasetMetadata>), Status> {
        let dataset: RemoteDataset = serde_json::from_str(&serialized_dataset).map_err(|e| {
            Status::invalid_argument(format!("Could not deserialize RemoteDataset: {}", e))
        })?;

        let dataset_metadata = self.datasets_metadata.read().unwrap();
        let dataset_metadata = if !dataset.identifier.is_empty() {
            let id = dataset.identifier;
            Some(
                dataset_metadata
                    .get(&id)
                    .ok_or(Status::aborted(format!("{:?} not found", id)))?
                    .clone(),
            )
        } else {
            None
        };

        let hash =
            hex::encode(digest::digest(&digest::SHA256, serialized_dataset.as_bytes()).as_ref());

        let mut samples_inputs = vec![];

        for input in dataset.inputs {
            samples_inputs.push(self.get_tensor(&input.identifier)?);
        }

        let labels = self.get_tensor(&dataset.labels.identifier)?;
        let limit = dataset.privacy_limit;

        let data = Dataset::new(samples_inputs, labels, limit);

        Ok((Arc::new(RwLock::new(data)), hash, dataset_metadata.clone()))
    }

    fn insert_dataset_remote_dataset_relation(
        &self,
        dataset_id: &str,
        remote_dataset_ref: &RemoteDatasetReference,
    ) {
        let mut relations = self.dataset_remote_dataset_relation.write().unwrap();

        relations.insert(dataset_id.to_string(), remote_dataset_ref.clone());
    }
    fn convert_from_dataset_to_remote_dataset(
        &self,
        dataset: &mut Dataset,
        metadata: &DatasetMetadata,
    ) -> Result<(RemoteDatasetReference, bool), Status> {
        let mut samples_locks = dataset
            .samples_inputs
            .iter()
            .map(|tensor| {
                let tensor = tensor.lock().unwrap();
                tensor
            })
            .collect::<Vec<_>>();
        let create_tensor = |tensor: Tensor| -> Reference {
            let meta = create_tensor_meta(&tensor);

            let identifier = self.insert_tensor(tensor);

            Reference {
                identifier,
                meta: meta.encode_to_vec(),
                name: format!("Tensor-{}", metadata.name.clone()),
                ..Default::default()
            }
        };

        let mut inputs = vec![];

        let labels_lock = dataset.labels.lock().unwrap();

        for sample in samples_locks.drain(..) {
            inputs.push(create_tensor(sample.detach()));
        }
        let labels = create_tensor(labels_lock.detach());
        let single_tensor = inputs.is_empty();

        Ok((
            RemoteDatasetReference {
                identifier: Some(metadata.clone().identifier),
                inputs,
                labels: Some(labels),
            },
            single_tensor,
        ))
    }
}

#[tonic::async_trait]
impl TorchService for BastionLabTorch {
    type FetchDatasetStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchModuleStream = ReceiverStream<Result<Chunk, Status>>;

    async fn send_dataset(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<RemoteDatasetReference>, Status> {
        let token = self.sess_manager.verify_request(&request)?;
        let client_info = self.sess_manager.get_client_info(token)?;
        let start_time = Instant::now();

        let artifact: Artifact<SizedObjectsBytes> = unstream_data(request.into_inner()).await?;
        let name = { artifact.name.clone() };
        let (dataset_hash, dataset_size) = {
            let lock = artifact.data.read().unwrap();
            let data = lock.get();
            let hash = hex::encode(digest::digest(&digest::SHA256, &data).as_ref());
            (hash, data.len())
        };

        let dataset: Artifact<Dataset> = tcherror_to_status((artifact).deserialize())?;

        // DatasetMetadata object is just a light-copy of the metadata sent with dataset and
        // stored in Artifact.
        // This is not be confused with the `meta` field which actually stores the meta of the Dataset
        // including the `nb_samples`, `privacy_limit`, and other dataset specific feature.
        let dataset_metadata = self.insert_dataset_metadata(&dataset);

        let (remote_dataset, single_tensor) = {
            let mut dataset = dataset.data.write().unwrap();

            self.convert_from_dataset_to_remote_dataset(&mut dataset, &dataset_metadata)?
        };

        if !single_tensor {
            // Here, we create the relationship between `Dataset` and `RemoteDataset` by adding them to
            // `dataset_remote_dataset_relation` storage in the database.

            // We do this because if we would want to avoid doubling the tensor storage (i.e., storing
            // tensors in `BastionLabTorchState.tensors` and `BastionLabTorchState.dataset`, then we would have to
            // manually create that relationship so that when a `fetch_dataset` API call is made, we can reconstruct the real `Dataset`
            // from the references of `RemoteTensor`.)

            self.insert_dataset_remote_dataset_relation(
                &dataset_metadata.identifier,
                &remote_dataset,
            );

            let elapsed = start_time.elapsed();
            info!(
                "Successfully uploaded Dataset {} in {}ms",
                dataset_metadata.identifier,
                elapsed.as_millis()
            );

            telemetry::add_event(
                TelemetryEventProps::SendDataset {
                    dataset_name: Some(name.clone()),
                    dataset_size,
                    time_taken: elapsed.as_millis() as f64,
                    dataset_hash: Some(dataset_hash.clone()),
                },
                Some(client_info),
            );
        }
        Ok(Response::new(remote_dataset))
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
            let model_hash = Uuid::new_v4().to_string();
            (model_hash, data.len())
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

        info!(
            "Successfully uploaded Model {} in {}ms",
            model_hash.clone(),
            elapsed.as_millis()
        );

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
        let (dataset, _, metadata) = {
            let identifier = request.into_inner().identifier;

            // Here, we use the dataset identifier from the request to read the relationship

            // We then convert the RemoteDatasetRef into a real RemoteDataset which contains references of RemoteTensors and
            // has no information about privacy_limit and nb_samples (we intentionally remove those parts because they are not
            // useful for fetching).

            // The goal is to effectively fetch the internal tensors related to the dataset.
            let read_lock = self.dataset_remote_dataset_relation.read().unwrap();
            let remote_dataset_ref = read_lock
                .get(&identifier)
                .ok_or(Status::failed_precondition(format!(
                    "Could not find dataset {identifier}"
                )))?
                .clone();
            let remote_dataset: RemoteDataset = remote_dataset_ref.into();

            let serialized_dataset = serde_json::to_string(&remote_dataset)
                .map_err(|e| Status::aborted(format!("Could not serialize RemoteDataset: {e}")))?;
            self.convert_from_remote_dataset(&serialized_dataset)?
        };

        let metadata = metadata.ok_or(Status::aborted("Cannot fetch a Lazy RemoteDataset. Please use the RemoteTensor.fetch API to fetch individual tensors"))?;
        let artifact = Artifact {
            data: dataset,
            description: metadata.description,
            name: metadata.name,
            meta: metadata.meta,
            client_info: None,
            secret: Key::new(ring::hmac::HMAC_SHA256, &[0]),
        };
        let serialized = tcherror_to_status(artifact.serialize())?;

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
        // Now that we have individual tensor access, it means that if a dataset is deleted, all the related tensors
        // **must** be deleted as well.

        // We use the [`RemoteDataset`] object which contains identifiers to all the related tensors of a particular
        // dataset to delete all the related tensors.

        // Also, we will delete the relationship between Dataset and RemoteDatasetRef

        let mut tensors = self.tensors.write().unwrap();
        let mut dataset_metadata = self.datasets_metadata.write().unwrap();
        let mut dataset_remote_dataset_relations =
            self.dataset_remote_dataset_relation.write().unwrap();

        let identifier = request.into_inner().identifier;

        // This is an on-the-fly converted RemoteDataset from a stored relation.
        let dataset: RemoteDataset = dataset_remote_dataset_relations
            .get(&identifier)
            .ok_or(Status::aborted(format!(
                "Dataset not found: {:?}",
                identifier
            )))?
            .clone()
            .into();

        // Deletes all associated inputs tensors from the database
        dataset.inputs.iter().for_each(|t| {
            tensors.remove(&t.identifier);
        });

        // Deletes labels associated tensor from the database
        tensors.remove(&dataset.labels.identifier);

        // Removes all relationships of the Dataset (DatasetMeta and DatasetRemoteSetRelations)
        if !dataset.identifier.is_empty() {
            dataset_metadata.remove(&dataset.identifier);
            dataset_remote_dataset_relations.remove(&dataset.identifier);
        }
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

        let (dataset, dataset_id, _) = self.convert_from_remote_dataset(&config.dataset)?;

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

        let (dataset, dataset_id, _) = self.convert_from_remote_dataset(&config.dataset)?;

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
        let list: Vec<Reference> = self
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
        let list: Vec<Reference> = self
            .datasets_metadata
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

    async fn send_tensor(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<Reference>, Status> {
        let res = unstream_data(request.into_inner()).await?;

        let (tensor, meta) = {
            let data = res.data.read().unwrap();
            let data: Tensor = (&*data).try_into().unwrap();
            let meta = create_tensor_meta(&data);
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

    async fn modify_tensor(
        &self,
        request: Request<UpdateTensor>,
    ) -> Result<Response<Reference>, Status> {
        let mut tensors = self.tensors.write().unwrap();

        let (identifier, dtype) = (&request.get_ref().identifier, &request.get_ref().dtype);

        let tensor = tensors
            .get_mut(identifier)
            .ok_or(Status::not_found("Could not find tensor"))?;

        let mut locked_tensor = tensor.lock().unwrap();

        *locked_tensor = locked_tensor.to_dtype(get_kind(&dtype)?, true, true);

        let meta = TensorMetaData {
            input_dtype: vec![format!("{:?}", locked_tensor.kind())],
            input_shape: locked_tensor.size(),
        };
        Ok(Response::new(Reference {
            identifier: identifier.clone(),
            name: String::new(),
            description: String::new(),
            meta: meta.encode_to_vec(),
        }))
    }
}
