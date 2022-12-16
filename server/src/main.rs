use bastionlab_common::config::BastionLabConfig;
use bastionlab_common::prelude::*;
use bastionlab_common::{
    auth::KeyManagement,
    session::SessionManager,
    telemetry::{self, TelemetryEventProps},
};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::path::Path;
use tonic::transport::{Identity, Server, ServerTlsConfig};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let config: BastionLabConfig =
        toml::from_str(&fs::read_to_string("config.toml").context("Reading the config.toml file")?)
            .context("Parsing the config.toml file")?;

    let disable_authentication = !std::env::var("DISABLE_AUTHENTICATION").is_err();

    let keys = if !disable_authentication {
        match KeyManagement::load_from_dir(Path::new(
            &config
                .public_keys_directory()
                .context("Parsing the public_keys_directory config path")?,
        )) {
            Ok(keys) => {
                info!("Authentication is enabled.");
                Some(keys)
            }
            Err(e) => {
                println!("Exiting due to an error reading keys. {}", e.message());
                //Temp fix to exit early, returning an error seems to break the "?" handlers above.
                return Ok(());
            }
        }
    } else {
        info!("Authentication is disabled.");
        None
    };

    let sess_manager = Arc::new(SessionManager::new(
        keys,
        config
            .session_expiry()
            .context("Parsing the public session_expiry config")?,
    ));

    let server_cert =
        fs::read("tls/host_server.pem").context("Reading the tls/host_server.pem file")?;
    let server_key =
        fs::read("tls/host_server.key").context("Reading the tls/host_server.key file")?;
    let server_identity = Identity::from_pem(&server_cert, &server_key);

    //TODO: Change it when specifying the TEE will be available
    let tee_mode = String::from("None");
    let platform: String = String::from(format!("{} - TEE Mode: {}", whoami::platform(), tee_mode));
    let uid: String = {
        let mut hasher = DefaultHasher::new();
        whoami::username().hash(&mut hasher);
        whoami::hostname().hash(&mut hasher);
        platform.hash(&mut hasher);
        String::from(format!("{:X}", hasher.finish()))
    };

    if std::env::var("BASTIONLAB_DISABLE_TELEMETRY").is_err() {
        telemetry::setup(platform, uid, tee_mode).context("Setting up telemetry")?;
        info!("Telemetry is enabled.")
    } else {
        info!("Telemetry is disabled.")
    }
    telemetry::add_event(TelemetryEventProps::Started {}, None);

    let mut builder = Server::builder()
        .tls_config(ServerTlsConfig::new().identity(server_identity))
        .context("Setting up TLS")?;

    // Session
    let builder = {
        use bastionlab_common::{
            session::SessionGrpcService,
            session_proto::session_service_server::SessionServiceServer,
        };
        let svc = SessionGrpcService::new(sess_manager.clone());
        builder.add_service(SessionServiceServer::new(svc))
    };

    // Polars
    let builder = {
        use bastionlab_polars::{
            polars_proto::polars_service_server::PolarsServiceServer, BastionLabPolars,
        };
        let svc = BastionLabPolars::new(sess_manager.clone());
        match BastionLabPolars::load_dfs(&svc) {
            Ok(_) => info!("Successfully loaded saved dataframes"),
            Err(_) => info!("There was an error loading saved dataframes"),
        };
        builder.add_service(PolarsServiceServer::new(svc))
    };

    // Torch
    let builder = {
        use bastionlab_torch::{
            torch_proto::torch_service_server::TorchServiceServer, BastionLabTorch,
        };
        let svc = BastionLabTorch::new(sess_manager.clone());
        builder.add_service(TorchServiceServer::new(svc))
    };

    let addr = config
        .client_to_enclave_untrusted_socket()
        .context("Parsing the client_to_enclave_untrusted_socket config")?;

    info!("BastionLab server listening on {addr:?}.");
    info!("Server ready to take requests");

    // serve!
    builder.serve(addr).await?;
    Ok(())
}
