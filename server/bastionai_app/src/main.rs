use env_logger::Env;
use log::info;
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::{fs::File, io::Read};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Identity;
use tonic::transport::ServerTlsConfig;

use tonic::{transport::Server, Request, Response, Status, Streaming};
use uuid::Uuid;

use bastionai_learning::{data::Dataset, nn::Module};

mod remote_torch {
    tonic::include_proto!("remote_torch");
}
use remote_torch::remote_torch_server::{RemoteTorch, RemoteTorchServer};
use remote_torch::{
    Chunk, Devices, Empty, Metric, Optimizers, Reference, References, TestConfig, TrainConfig
};

mod storage;
use storage::Artifact;

mod utils;
use utils::*;

mod learning;
use learning::*;

mod serialization;
use serialization::*;

struct BastionAIServer {
    modules: RwLock<HashMap<Uuid, Artifact<Module>>>,
    datasets: RwLock<HashMap<Uuid, Artifact<Dataset>>>,
    runs: RwLock<HashMap<Uuid, Arc<RwLock<Run>>>>,
}

impl BastionAIServer {
    pub fn new() -> Self {
        BastionAIServer {
            modules: RwLock::new(HashMap::new()),
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

        let dataset: Artifact<Dataset> =
            tcherror_to_status((unstream_data(request.into_inner()).await?).deserialize())?;
        let name = String::from(dataset.name.clone());
        let description = String::from(dataset.description.clone());
        let identifier = Uuid::new_v4();

        self.datasets
            .write()
            .unwrap()
            .insert(identifier.clone(), dataset);

        let elapsed = start_time.elapsed();
        info!("Upload Dataset successful in {}ms", elapsed.as_millis());

        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name,
            description,
        }))
    }

    async fn send_model(
        &self,
        request: Request<Streaming<Chunk>>,
    ) -> Result<Response<Reference>, Status> {
        let start_time = Instant::now();

        let module: Artifact<Module> =
            tcherror_to_status(unstream_data(request.into_inner()).await?.deserialize())?;
        let name = String::from(module.name.clone());
        let description = String::from(module.description.clone());
        let identifier = Uuid::new_v4();

        self.modules
            .write()
            .unwrap()
            .insert(identifier.clone(), module);
        let elapsed = start_time.elapsed();
        info!("Upload Model successful in {}ms", elapsed.as_millis());

        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name,
            description,
        }))
    }

    async fn fetch_dataset(
        &self,
        request: Request<Reference>,
    ) -> Result<Response<Self::FetchDatasetStream>, Status> {
        let identifier = parse_reference(request.into_inner())?;
        let serialized = {
            let datasets = self.datasets.read().unwrap();
            let artifact = datasets
                .get(&identifier)
                .ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 100_000_000).await)
    }

    async fn fetch_module(
        &self,
        request: Request<Reference>,
    ) -> Result<Response<Self::FetchModuleStream>, Status> {
        let identifier = parse_reference(request.into_inner())?;
        let serialized = {
            let modules = self.modules.read().unwrap();
            let artifact = modules
                .get(&identifier)
                .ok_or(Status::not_found("Not found"))?;
            tcherror_to_status(artifact.serialize())?
        };

        Ok(stream_data(serialized, 100_000_000).await)
    }

    async fn delete_dataset(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = parse_reference(request.into_inner())?;
        self.datasets.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn delete_module(&self, request: Request<Reference>) -> Result<Response<Empty>, Status> {
        let identifier = parse_reference(request.into_inner())?;
        self.modules.write().unwrap().remove(&identifier);
        Ok(Response::new(Empty {}))
    }

    async fn train(&self, request: Request<TrainConfig>) -> Result<Response<Reference>, Status> {
        let config = request.into_inner();
        let dataset_id = parse_reference(
            config
                .dataset
                .clone()
                .ok_or(Status::invalid_argument("Not found"))?,
        )?;
        let module_id = parse_reference(
            config
                .model
                .clone()
                .ok_or(Status::invalid_argument("Not found"))?,
        )?;
        let device = parse_device(&config.device)?;
        let module = {
            let modules = self.modules.read().unwrap();
            let module = modules
                .get(&module_id)
                .ok_or(Status::not_found("Not found"))?;
            Arc::clone(&module.data)
        };
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            let dataset = datasets
                .get(&dataset_id)
                .ok_or(Status::not_found("Not found"))?;
            Arc::clone(&dataset.data)
        };

        let identifier = Uuid::new_v4();
        self.runs
            .write()
            .unwrap()
            .insert(identifier, Arc::new(RwLock::new(Run::Pending)));
        let run = Arc::clone(self.runs.read().unwrap().get(&identifier).unwrap());
        module_train(module, dataset, run, config, device);
        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name: format!("Run #{}", identifier),
            description: String::from(""),
        }))
    }

    async fn test(&self, request: Request<TestConfig>) -> Result<Response<Reference>, Status> {
        let config = request.into_inner();
        let dataset_id = parse_reference(
            config
                .dataset
                .clone()
                .ok_or(Status::invalid_argument("Not found"))?,
        )?;
        let module_id = parse_reference(
            config
                .model
                .clone()
                .ok_or(Status::invalid_argument("Not found"))?,
        )?;
        let device = parse_device(&config.device)?;
        let module = {
            let modules = self.modules.read().unwrap();
            let module = modules
                .get(&module_id)
                .ok_or(Status::not_found("Not found"))?;
            Arc::clone(&module.data)
        };
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            let dataset = datasets
                .get(&dataset_id)
                .ok_or(Status::not_found("Not found"))?;
            Arc::clone(&dataset.data)
        };

        let identifier = Uuid::new_v4();
        self.runs
            .write()
            .unwrap()
            .insert(identifier, Arc::new(RwLock::new(Run::Pending)));
        let run = Arc::clone(self.runs.read().unwrap().get(&identifier).unwrap());
        module_test(module, dataset, run, config, device);
        Ok(Response::new(Reference {
            identifier: format!("{}", identifier),
            name: format!("Run #{}", identifier),
            description: String::from(""),
        }))
    }

    async fn available_models(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<References>, Status> {
        let list = self
            .modules
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| Reference {
                identifier: format!("{}", k),
                name: v.name.clone(),
                description: v.description.clone(),
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
        let identifier = parse_reference(request.into_inner())?;

        match &*self
            .runs
            .read()
            .unwrap()
            .get(&identifier)
            .unwrap()
            .read()
            .unwrap()
        {
            Run::Pending => Err(Status::internal("Run has not started.")),
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

    let server = BastionAIServer::new();

    let mut file = File::open("config.toml")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let network_config: bastionai_common::NetworkConfig = toml::from_str(&contents)?;

    info!(
        "BastionAI listening on {}",
        network_config.client_to_enclave_untrusted_socket()?
    );
    Server::builder()
        .tls_config(ServerTlsConfig::new().identity(server_identity))?
        .add_service(RemoteTorchServer::new(server))
        .serve(network_config.client_to_enclave_untrusted_socket()?)
        .await?;

    Ok(())
}
