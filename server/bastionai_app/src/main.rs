use bastionai_learning::nn::Module;
use env_logger::Env;
use log::info;
use std::collections::HashMap;
use std::{
    hash::{Hash, Hasher},
    collections::hash_map::DefaultHasher};
use std::ffi::CString;
use std::fs;
use std::sync::{Arc, RwLock, Mutex};
use std::time::Instant;
use std::{fs::File, io::Read};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Identity;
use tonic::transport::ServerTlsConfig;
use anyhow::Context;

use ring::{digest, rand};

use tonic::{transport::Server, Request, Response, Status, Streaming};

use bastionai_learning::{data::Dataset, nn::CheckPoint};

mod remote_torch {
    tonic::include_proto!("remote_torch");
}
use remote_torch::remote_torch_server::{RemoteTorch, RemoteTorchServer};
use remote_torch::{
    Chunk, ClientInfo, ListResponse, Empty, MetricResponse, ReferenceRequest, ReferenceResponse, ReferenceListResponse, TestRequest,
    TrainRequest, ChallengeResponse
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

mod access_control;

use bastionai_learning::serialization::{SizedObjectsBytes, BinaryModule};
use bastionai_common::auth::{auth_interceptor, setup_jwt};

/// The server's state
struct BastionAIServer {
    binaries: RwLock<HashMap<Vec<u8>, Artifact<BinaryModule>>>,
    checkpoints: RwLock<HashMap<Vec<u8>, Artifact<CheckPoint>>>,
    datasets: RwLock<HashMap<Vec<u8>, Artifact<Dataset>>>,
    runs: RwLock<HashMap<Vec<u8>, Artifact<Run>>>,
    challenges: Mutex<Vec<[u8; 32]>>,
}

impl BastionAIServer {
    pub fn new() -> Self {
        BastionAIServer {
            binaries: RwLock::new(HashMap::new()),
            checkpoints: RwLock::new(HashMap::new()),
            datasets: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
            challenges: Mutex::new(Vec::new()),
        }
    }

    fn check_challenge<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if let Some(meta) = request.metadata().get_bin("challenge") {
            let challenge = meta.to_bytes().map_err(|_| {
                Status::invalid_argument("Could not decode challenge")
            })?;
            let lock = self.challenges.lock().unwrap();
            for c in lock.iter().rev() {
                if c == &*challenge {
                    return Ok(())
                }
            }
            Err(Status::permission_denied("Invalid or reused challenge"))
        } else {
            Ok(())
        }
    }
}

#[tonic::async_trait]
impl RemoteTorch for BastionAIServer {
    type FetchDatasetStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchModelStream = ReceiverStream<Result<Chunk, Status>>;
    type FetchCheckpointStream = ReceiverStream<Result<Chunk, Status>>;

    async fn send_dataset(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        let artifact: Artifact<SizedObjectsBytes> = unstream_data(request.into_inner()).await?;

        let (dataset_hash, dataset_size) = {
            let lock = artifact.data.read().unwrap();
            let data = lock.get();
            let hash = digest::digest(&digest::SHA256, &data).as_ref().to_vec();
            (hash, data.len())
        };

        let dataset: Artifact<Dataset> = tcherror_to_status((artifact).deserialize())?;
        let name = dataset.name.clone();
        let description = dataset.description.clone();
        let license = dataset.license.clone();
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
                dataset_hash: Some(hex::encode(&dataset_hash))
            },
            client_info,
        );
        Ok(Response::new(ReferenceResponse {
            hash: dataset_hash,
            name,
            description,
            license: serde_json::to_string(&license).unwrap(),
            meta,
        }))
    }

    async fn send_model(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<ReferenceResponse>, Status> {
        let start_time = Instant::now();

        let artifact: Artifact<SizedObjectsBytes> = unstream_data(request.into_inner()).await?;
        
        let (model_hash, model_size) = {
            let lock = artifact.data.read().unwrap();
            let data = lock.get();
            let hash = digest::digest(&digest::SHA256, &data).as_ref().to_vec();
            (hash, data.len())
        };
        
        let binary = tcherror_to_status(artifact.deserialize())?;
            
        let name = binary.name.clone();
        let description = binary.description.clone();
        let license = binary.license.clone();
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
                model_hash: Some(hex::encode(&model_hash)),
                model_size,
                time_taken: elapsed.as_millis() as f64,
            },
            client_info,
        );
        Ok(Response::new(ReferenceResponse {
            hash: model_hash,
            name,
            description,
            license: serde_json::to_string(&license).unwrap(),
            meta,
        }))
    }

    async fn fetch_dataset(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchDatasetStream>, Status> {
        self.check_challenge(&request)?;
        let serialized = {
            let datasets = self.datasets.read().unwrap();
            let artifact = datasets
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Dataset not found"))?;
            artifact.license.verify_fetch(&request)?;
            
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 4_194_285, "Dataset".to_string()).await)
    }

    async fn fetch_model(&self, request: Request<ReferenceRequest>) -> Result<Response<Self::FetchModelStream>, Status> {
        self.check_challenge(&request)?;
        let serialized = {
            let binaries = self.binaries.read().unwrap();
            let artifact = binaries
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Model not found"))?;
            artifact.license.verify_fetch(&request)?;
            
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 4_194_285, "Model".to_string()).await)
    }

    async fn fetch_checkpoint(
        &self,
        request: Request<ReferenceRequest>,
    ) -> Result<Response<Self::FetchCheckpointStream>, Status> {
        self.check_challenge(&request)?;
        let serialized = {
            let checkpoints = self.checkpoints.read().unwrap();
            
            let checkpoint = checkpoints
                .get(&request.get_ref().hash);
            match checkpoint {
                Some(artifact) => {
                    artifact.license.verify_fetch(&request)?;
                    let checkpoints = &artifact.data.read().unwrap().data;
                    let last_chkpt = &checkpoints[checkpoints.len() - 1];
        
                    let mut chkpt_bytes = SizedObjectsBytes::new();
                    chkpt_bytes.append_back(last_chkpt.clone());
        
                    Artifact {
                        data: Arc::new(RwLock::new(chkpt_bytes)),
                        name: artifact.name.clone(),
                        client_info: artifact.client_info.clone(),
                        description: artifact.description.clone(),
                        meta: artifact.meta.clone(),
                        license: artifact.license.clone(),
                    }
                }
                None => {
                    let binaries = self.binaries.read().unwrap();
                    let binary = binaries.get(&request.get_ref().hash).ok_or(Status::not_found("Module not found!"))?;
                    binary.license.verify_fetch(&request)?;
                    let module: Module = (&*binary.data.read().unwrap()).try_into().unwrap();
                    let module = Artifact {
                        data: Arc::new(RwLock::new(module)),
                        name: binary.name.clone(),
                        client_info: binary.client_info.clone(),
                        description: binary.description.clone(),
                        meta: binary.meta.clone(),
                        license: binary.license.clone(),
                    };
                    tcherror_to_status(module.serialize())?

                }
            }
        };

        Ok(stream_data(serialized, 4_194_285, "Model".to_string()).await)
    }

    async fn fetch_run(&self, request: Request<ReferenceRequest>) -> Result<Response<ReferenceResponse>, Status> {
        self.check_challenge(&request)?;
        let runs = self.runs.read().unwrap();
        let artifact = runs
            .get(&request.get_ref().hash)
            .ok_or(Status::not_found("Run not found"))?;
        artifact.license.verify_fetch(&request)?;

        Ok(Response::new(ReferenceResponse {
            hash: request.into_inner().hash,
            name: String::new(),
            description: String::new(),
            license: serde_json::to_string(&artifact.license).unwrap(),
            meta: Vec::new(),
        }))
    }

    async fn delete_dataset(&self, request: Request<ReferenceRequest>) -> Result<Response<Empty>, Status> {
        self.check_challenge(&request)?;
        {
            let mut datasets = self.datasets.write().unwrap();
            let artifact = datasets
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Dataset not found"))?;
            artifact.license.verify_delete(&request)?;
            datasets.remove(&request.get_ref().hash);
        }
        Ok(Response::new(Empty {}))
    }

    async fn delete_model(&self, request: Request<ReferenceRequest>) -> Result<Response<Empty>, Status> {
        self.check_challenge(&request)?;
        {
            let mut binaries = self.binaries.write().unwrap();
            let binary = binaries
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Model not found"))?;
            binary.license.verify_delete(&request)?;
            binaries.remove(&request.get_ref().hash);
        }
        Ok(Response::new(Empty {}))
    }

    async fn delete_checkpoint(&self, request: Request<ReferenceRequest>) -> Result<Response<Empty>, Status> {
        self.check_challenge(&request)?;
        {
            let mut checkpoints = self.checkpoints.write().unwrap();
            let checkpoint = checkpoints
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Model not found"))?;
            checkpoint.license.verify_delete(&request)?;
            checkpoints.remove(&request.get_ref().hash);
        }
        Ok(Response::new(Empty {}))
    }

    async fn delete_run(&self, request: Request<ReferenceRequest>) -> Result<Response<Empty>, Status> {
        self.check_challenge(&request)?;
        {
            let mut runs = self.runs.write().unwrap();
            let run = runs
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Model not found"))?;
            run.license.verify_delete(&request)?;
            runs.remove(&request.get_ref().hash);
        }
        Ok(Response::new(Empty {}))
    }

    async fn train(&self, request: Request<TrainRequest>) -> Result<Response<ReferenceResponse>, Status> {
        self.check_challenge(&request)?;
        let config = request.get_ref();
        let model_hash = config.model.clone();
        let dataset_hash = config.dataset.clone();
        let device = parse_device(&config.device)?;
        
        let (binary, checkpoint, dataset, license) = {
            let binaries = self.binaries.read().unwrap();
            let mut checkpoints = self.checkpoints.write().unwrap();
            let datasets = self.datasets.read().unwrap();
            
            let binary = binaries
                .get(&model_hash)
                .ok_or(Status::not_found("Model not found"))?;
            
            let checkpoint = if config.resume {
                checkpoints.get(&model_hash).ok_or(Status::not_found("CheckPoint not found!"))?
            } else {
                let binaries = self.binaries.read().unwrap();
                let binary = binaries
                    .get(&model_hash)
                    .ok_or(Status::not_found("Model not found"))?;
                let chkpt = Artifact {
                    data: Arc::new(RwLock::new(CheckPoint::new(config.eps >= 0.0))),
                    name: binary.name.clone(),
                    client_info: binary.client_info.clone(),
                    description: binary.description.clone(),
                    meta: binary.meta.clone(),
                    license: binary.license.clone(),
                };
                checkpoints.insert(model_hash.clone(), chkpt);
                checkpoints.get(&model_hash).unwrap()
            };

            let dataset = datasets
                .get(&dataset_hash)
                .ok_or(Status::not_found("Dataset not found"))?;
            
            binary.license.verify_train(&request)?;
            checkpoint.license.verify_train(&request)?;
            dataset.license.verify_train(&request)?;

            (
                Arc::clone(&binary.data),
                Arc::clone(&checkpoint.data),
                Arc::clone(&dataset.data),
                binary.license.combine(&checkpoint.license)?.combine(&dataset.license)?
            )
        };

        let identifier: [u8; 32] = rand::generate(&rand::SystemRandom::new()).map_err(|_| Status::internal("Could not generate random value"))?.expose();
        self.runs
            .write()
            .unwrap()
            .insert(identifier.clone().into(), Artifact {
                data: Arc::new(RwLock::new(Run::new(RunConfig::Train(request.get_ref().clone())))),
                name: String::new(),
                description: String::new(),
                license: license.clone(),
                meta: Vec::new(),
                client_info: request.get_ref().client_info.clone(),
            });
        let run = Arc::clone(&self.runs.read().unwrap().get(&identifier[..]).unwrap().data);
        module_train(binary, dataset, checkpoint, run, request.into_inner(), device);
        Ok(Response::new(ReferenceResponse {
            hash: identifier.into(),
            name: String::new(),
            description: String::new(),
            license: serde_json::to_string(&license).unwrap(),
            meta: Vec::new(),
        }))
    }

    async fn test(&self, request: Request<TestRequest>) -> Result<Response<ReferenceResponse>, Status> {
        self.check_challenge(&request)?;
        let config = request.get_ref();
        let model_hash = config.model.clone();
        let dataset_hash = config.dataset.clone();
        let device = parse_device(&config.device)?;
        
        let (binary, checkpoint, dataset, license) = {
            let binaries = self.binaries.read().unwrap();
            let checkpoints = self.checkpoints.read().unwrap();
            let datasets = self.datasets.read().unwrap();

            let binary = binaries
                .get(&model_hash)
                .ok_or(Status::not_found("Model not found"))?;
            let checkpoint = checkpoints
                .get(&model_hash)
                .ok_or(Status::not_found("Checkpoint not found"))?;
            let dataset = datasets
                .get(&dataset_hash)
                .ok_or(Status::not_found("Dataset not found"))?;
                
            checkpoint.license.verify_test(&request)?;
            binary.license.verify_test(&request)?;
            dataset.license.verify_test(&request)?;

            (
                Arc::clone(&binary.data),
                Arc::clone(&checkpoint.data),
                Arc::clone(&dataset.data),
                binary.license.combine(&checkpoint.license)?.combine(&dataset.license)?
            )
        };

        let identifier: [u8; 32] = rand::generate(&rand::SystemRandom::new()).map_err(|_| Status::internal("Could not generate random value"))?.expose();
        self.runs
            .write()
            .unwrap()
            .insert(identifier.to_vec(), Artifact {
                data: Arc::new(RwLock::new(Run::new(RunConfig::Test(request.get_ref().clone())))),
                name: String::new(),
                description: String::new(),
                license: license.clone(),
                meta: Vec::new(),
                client_info: request.get_ref().client_info.clone(),
            });
        let run = Arc::clone(&self.runs.read().unwrap().get(&identifier[..]).unwrap().data);
        module_test(binary, dataset, checkpoint, run, request.into_inner(), device);
        Ok(Response::new(ReferenceResponse {
            hash: identifier.into(),
            name: String::new(),
            description: String::new(),
            license: serde_json::to_string(&license).unwrap(),
            meta: Vec::new(),
        }))
    }

    async fn available_models(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceListResponse>, Status> {
        self.check_challenge(&request)?;
        let list = self
            .binaries
            .read()
            .unwrap()
            .iter()
            .filter(|(_, artifact)| artifact.license.verify_list(&request).is_ok())
            .map(|(k, v)| ReferenceResponse {
                hash: k.clone(),
                name: v.name.clone(),
                description: v.description.clone(),
                license: serde_json::to_string(&v.license).unwrap(),
                meta: v.meta.clone(),
            })
            .collect();

        Ok(Response::new(ReferenceListResponse { list }))
    }

    async fn available_checkpoints(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceListResponse>, Status> {
        self.check_challenge(&request)?;
        let list = self
            .checkpoints
            .read()
            .unwrap()
            .iter()
            .filter(|(_, artifact)| artifact.license.verify_list(&request).is_ok())
            .map(|(k, v)| ReferenceResponse {
                hash: k.clone(),
                name: v.name.clone(),
                description: v.description.clone(),
                license: serde_json::to_string(&v.license).unwrap(),
                meta: v.meta.clone(),
            })
            .collect();

        Ok(Response::new(ReferenceListResponse { list }))
    }

    async fn available_datasets(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<ReferenceListResponse>, Status> {
        self.check_challenge(&request)?;
        let list = self
            .datasets
            .read()
            .unwrap()
            .iter()
            .filter(|(_, artifact)| artifact.license.verify_list(&request).is_ok())
            .map(|(k, v)| ReferenceResponse {
                hash: k.clone(),
                name: v.name.clone(),
                description: v.description.clone(),
                license: serde_json::to_string(&v.license).unwrap(),
                meta: v.meta.clone(),
            })
            .collect();

        Ok(Response::new(ReferenceListResponse { list }))
    }

    async fn available_devices(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ListResponse>, Status> {
        let mut list = vec![String::from("cpu")];
        if tch::Cuda::is_available() {
            list.push(String::from("gpu"));
            for index in 0..tch::Cuda::device_count() {
                list.push(format!("cuda:{}", index));
            }
        }

        Ok(Response::new(ListResponse { list }))
    }

    async fn available_optimizers(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ListResponse>, Status> {
        let list = vec!["SGD", "Adam"].iter().map(|v| v.to_string()).collect();
        Ok(Response::new(ListResponse { list }))
    }

    async fn get_metric(&self, request: Request<ReferenceRequest>) -> Result<Response<MetricResponse>, Status> {
        self.check_challenge(&request)?;
        let metric = {
            let runs = self.runs.read().unwrap();
            let run = runs
                .get(&request.get_ref().hash)
                .ok_or(Status::not_found("Module not found"))?;
            
            
            // let run = Arc::clone(&runs
            //     .get(&request.get_ref().hash)
            //     .ok_or(Status::not_found("Module not found"))?
            //     .data
            // );
            run.license.verify_fetch(&request)?;

            let x = &run.data.read().unwrap().status;
            match x {
                RunStatus::Pending => return Err(Status::out_of_range("Run has not started.")),
                RunStatus::Ok(m) => m.clone(),
                RunStatus::Error(e) => return Err(Status::internal(e.message())),
            }
        };
        
        Ok(Response::new(metric))
    }

    async fn get_challenge(&self, _request: Request<Empty>) -> Result<Response<ChallengeResponse>, Status> {
        let rng = rand::SystemRandom::new();
        let challenge: [u8; 32] = rand::generate(&rng).map_err(|_| Status::internal("Could not generate random value"))?.expose();
        self.challenges.lock().unwrap().push(challenge);

        Ok(Response::new(ChallengeResponse { value: Vec::from(challenge) }))
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
    let server_cert = fs::read("./tls/host_server.pem").context("Reading file ./tls/host_server.pem")?;
    let server_key = fs::read("./tls/host_server.key").context("Reading file ./tls/host_server.key")?;
    let server_identity = Identity::from_pem(&server_cert, &server_key);

    setup_jwt();

    let server = BastionAIServer::new();

    let mut file = File::open("./config.toml").context("Reading file ./config.toml")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).context("Reading file ./config.toml")?;
    let network_config: bastionai_common::NetworkConfig = toml::from_str(&contents).context("Parsing file ./config.toml")?;

    let platform: CString = CString::new(format!("{}", whoami::platform())).unwrap();
    let uid: CString = {
        let mut hasher = DefaultHasher::new();
        whoami::username().hash(&mut hasher);
        whoami::hostname().hash(&mut hasher);
        platform.hash(&mut hasher);
        CString::new(format!("{:X}", hasher.finish())).unwrap()
    };

    if std::env::var("BASTIONAI_DISABLE_TELEMETRY").is_err() {
        telemetry::setup(platform.into_string().unwrap(), uid.into_string().unwrap()).context("Starting telemetry")?;
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
        .tls_config(ServerTlsConfig::new().identity(server_identity)).context("Configuring the TLS server identity")?
        .add_service(RemoteTorchServer::with_interceptor(server, auth_interceptor))
        .serve(network_config.client_to_enclave_untrusted_socket()?)
        .await.context("Running the server")?;

    Ok(())
}
