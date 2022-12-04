use std::{
    error::Error,
    fs::{self, File},
    io::Read,
};

use bastionlab_common::authentication::*;
use bastionlab_common::config::*;
use env_logger::Env;
use log::info;
use tonic::transport::Identity;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let mut file = File::open("config.toml")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: BastionLabConfig = toml::from_str(&contents)?;
    let keys = KeyManagement::load_from_dir(config.public_keys_directory()?)?;
    let server_cert = fs::read(String::from("tls/host_server.pem"))?;
    let server_key = fs::read(String::from("tls/host_server.key"))?;
    let server_identity = Identity::from_pem(&server_cert, &server_key);
    bastionlab_common::start(config, Some(keys), server_identity).await?;
    Ok(()) // Won't be reached
}
