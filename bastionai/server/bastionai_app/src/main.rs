use bastionai_learning::nn::Module;
use env_logger::Env;
use log::info;
use std::collections::HashMap;

use std::{
    hash::{Hash, Hasher},
    collections::hash_map::DefaultHasher};
use std::ffi::CString;
use std::fs;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::{fs::File, io::Read};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Identity;
use tonic::transport::ServerTlsConfig;

use ring::digest;

use tonic::{transport::Server, Request, Response, Status, Streaming};
use uuid::Uuid;

use bastionai_learning::{data::Dataset, nn::CheckPoint};

mod remote_torch {
    tonic::include_proto!("remote_torch");
}
use remote_torch::remote_torch_server::{RemoteTorch, RemoteTorchServer};
use remote_torch::{
    Chunk, ClientInfo, Devices, Empty, Metric, Optimizers, Reference, References, TestConfig,
    TrainConfig,
};

mod telemetry;
use telemetry::TelemetryEventProps;

mod storage;
use storage::Artifact;

mod utils;
use utils::*;

mod learning;
use learning::*;

mod serialization;
use serialization::*;

use bastionai_learning::serialization::{SizedObjectsBytes, BinaryModule};
use bastionai_common::auth::{auth_interceptor, setup_jwt};

/// The server's state
struct BastionAIServer {
    binaries: RwLock<HashMap<String, Artifact<BinaryModule>>>,
    checkpoints: RwLock<HashMap<String, Artifact<CheckPoint>>>,
    datasets: RwLock<HashMap<String, Artifact<Dataset>>>,
    runs: RwLock<HashMap<Uuid, Arc<RwLock<Run>>>>,
}

impl BastionAIServer {
    pub fn new() -> Self {
        BastionAIServer {
            binaries: RwLock::new(HashMap::new()),
            checkpoints: RwLock::new(HashMap::new()),
            datasets: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
        }
    }
}

#[tonic::async_trait]
impl RemoteTorch for BastionAIServer {
    type FetchDatasetStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchModuleStream = ReceiverStream<Result<Chunk, Status>>;

    async fn send_dataset(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<Reference>, Status> {
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
        let client_info = dataset.client_info.clone();

        self.datasets
            .write()
            .unwrap()
            .insert(dataset_hash.clone(), dataset);

        let elapsed = start_time.elapsed();
        info!(
        target: "BastionAI",
            
            "Upload Dataset successful in {}ms", elapsed.as_millis());

        telemetry::add_event(
            TelemetryEventProps::SendDataset {
                dataset_name: Some(name.clone()),
                dataset_size,
                time_taken: elapsed.as_millis() as f64,
                dataset_hash: Some(dataset_hash.clone())
            },
            client_info,
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
        let client_info = binary.client_info.clone();
        
        self.binaries.write().unwrap().insert(model_hash.clone(), binary);
        let elapsed = start_time.elapsed();

        info!(
        target: "BastionAI",
            
            "Upload Model successful in {}ms", elapsed.as_millis());

        telemetry::add_event(
            TelemetryEventProps::SendModel {
                model_name: Some(name.clone()),
                model_hash: Some(model_hash.clone()),
                model_size,
                time_taken: elapsed.as_millis() as f64,
            },
            client_info,
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
        let identifier = request.into_inner().identifier;
        
        let serialized = {
            let checkpoints = self.checkpoints.read().unwrap();
            
            let checkpoint = checkpoints
                .get(&identifier);
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
                        client_info: artifact.client_info.clone(),
                        secret: artifact.secret.clone(),
                        description: artifact.description.clone(),
                        meta: artifact.meta.clone()
                    }            
                }
                None => {
                    let binaries = self.binaries.read().unwrap();
                    let binary = binaries.get(&identifier).ok_or(Status::not_found("Module not found!"))?;
                    let module: Module = (&*binary.data.read().unwrap()).try_into().unwrap();
                    let module = Artifact {
                        data: Arc::new(RwLock::new(module)),
                        name: binary.name.clone(),
                        client_info: binary.client_info.clone(),
                        secret: binary.secret.clone(),
                        description: binary.description.clone(),
                        meta: binary.meta.clone()
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
        let config = request.into_inner();
        let dataset_id = config
            .dataset
            .clone()
            .ok_or(Status::invalid_argument("Invalid dataset reference"))?
            .identifier;
        let binary_id = config
            .model
            .clone()
            .ok_or(Status::invalid_argument("Invalid module reference"))?
            .identifier;
            let device = parse_device(&config.device)?;
        let (binary, chkpt, client_info) = {
            let binaries = self.binaries.read().unwrap();
            let binary: &Artifact<BinaryModule> = 
            binaries
            .get(&binary_id)
            .ok_or(Status::not_found("Module binary not found"))?; 
            let mut checkpoints = self.checkpoints.write().unwrap();
            let chkpt = if config.resume {

                let chkpt = checkpoints.get(&binary_id).ok_or(Status::not_found("CheckPoint not found!"))?;
                chkpt
            }else {
                let chkpt = Artifact {
                    data: Arc::new(RwLock::new(CheckPoint::new(config.eps >= 0.0))),
                    name: binary.name.clone(),
                    client_info: binary.client_info.clone(),
                    secret: binary.secret.clone(),
                    description: binary.description.clone(),
                    meta: binary.meta.clone(),
                };
            checkpoints.insert(binary_id.clone(), chkpt);
            let chkpt = checkpoints.get(&binary_id).ok_or(Status::not_found("Module binary not found"))?;
            chkpt
            };
            (Arc::clone(&binary.data), Arc::clone(&chkpt.data) , binary.client_info.clone())
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
        module_train(binary, dataset, run, config, device, binary_id, dataset_id, client_info, chkpt);
        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name: format!("Run #{}", identifier),
            description: String::from(""),
            meta: Vec::new(),
        }))
    }

    async fn test(&self, request: Request<TestConfig>) -> Result<Response<Reference>, Status> {
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
        let (module, binary, client_info) = {
            let chkpts_store = self.checkpoints.read().unwrap();
            let artifact = chkpts_store
                .get(&module_id)
                .ok_or(Status::not_found("Module not found"))?;
            let binaries = self.binaries.read().unwrap();
            let binary = binaries.get(&module_id).unwrap();
            
            (Arc::clone(&artifact.data), Arc::clone(&binary.data), artifact.client_info.clone())
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
        module_test(module,binary, dataset, run, config, device, module_id, dataset_id, client_info);
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {


    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let logo_str: &str = include_str!("../logo.txt");
    let version_str: String = format!("VERSION : {}", env!("CARGO_PKG_VERSION"));
    let text_size: usize = 58;
    println!("{}\n", logo_str);
    fill_blank_and_print("BastionAI - SECURE AI TRAINING SERVER", text_size);
    fill_blank_and_print("MADE BY MITHRIL SECURITY", text_size);
    fill_blank_and_print(
        "GITHUB: https://github.com/mithril-security/bastionai",
        text_size,
    );
    fill_blank_and_print(&version_str, text_size);

    // Identity for untrusted (non-attested) communication
    let server_cert = fs::read("tls/host_server.pem")?;
    let server_key = fs::read("tls/host_server.key")?;
    let server_identity = Identity::from_pem(&server_cert, &server_key);

    setup_jwt();

    let server = BastionAIServer::new();

    let mut file = File::open("config.toml")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let network_config: bastionai_common::NetworkConfig = toml::from_str(&contents)?;

    let platform: CString = CString::new(format!("{}", whoami::platform())).unwrap();
    let uid: CString = {
        let mut hasher = DefaultHasher::new();
        whoami::username().hash(&mut hasher);
        whoami::hostname().hash(&mut hasher);
        platform.hash(&mut hasher);
        CString::new(format!("{:X}", hasher.finish())).unwrap()
    };

    if std::env::var("BASTIONAI_DISABLE_TELEMETRY").is_err() {
        telemetry::setup(platform.into_string().unwrap(), uid.into_string().unwrap())?;
    }
    else {
        info!(
            target: "BastionAI",
            "Telemetry is disabled.")
    }
    telemetry::add_event(TelemetryEventProps::Started {}, None);
    info!(
        target: "BastionAI",
        "BastionAI listening on {}",
        network_config.client_to_enclave_untrusted_socket()?
    );


    Server::builder()
        .tls_config(ServerTlsConfig::new().identity(server_identity))?
        .add_service(RemoteTorchServer::with_interceptor(server, auth_interceptor))
        .serve(network_config.client_to_enclave_untrusted_socket()?)
        .await?;

    Ok(())
}
