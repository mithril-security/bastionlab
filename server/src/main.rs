use bastionlab_common::config::BastionLabConfig;
use bastionlab_common::prelude::*;
use bastionlab_common::{
    auth::KeyManagement,
    session::SessionManager,
    telemetry::{self, TelemetryEventProps},
};
use bastionlab_polars::BastionLabPolars;
use bastionlab_torch::BastionLabTorch;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::path::Path;
use std::time::SystemTime;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tonic::Status;

#[derive(Clone)]
struct TokenValidator {
    sess_manager: Arc<SessionManager>,
}

impl tonic::service::Interceptor for TokenValidator {
    fn call(&mut self, req: tonic::Request<()>) -> std::result::Result<tonic::Request<()>, Status> {
        if !self.sess_manager.auth_enabled() {
            return Ok(req);
        }
        let meta = req
            .metadata()
            .get_bin("accesstoken-bin")
            .ok_or_else(|| Status::invalid_argument("No access token in request metadata"))?;

        let access_token = meta
            .to_bytes()
            .map_err(|_| Status::invalid_argument("Could not decode accesstoken"))?;

        let mut tokens = self.sess_manager.sessions.write().expect("Poisoned lock");

        let session = tokens
            .get(access_token.as_ref())
            .ok_or_else(|| Status::aborted("Session not found!"))?;

        let recv_ip = &req
            .remote_addr()
            .ok_or_else(|| Status::aborted("User IP unavailable"))?;

        // ip verification
        if session.user_ip.ip() != recv_ip.ip() {
            return Err(Status::aborted("Unknown IP Address!"));
        }

        // expiry verification
        let curr_time = SystemTime::now();
        if curr_time > session.expiry {
            tokens.remove(access_token.as_ref());
            return Err(Status::aborted("Session Expired"));
        }

        Ok(req)
    }
}

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

    let sess_manager: Arc<SessionManager> = Arc::new(SessionManager::new(
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

    let token_validator = TokenValidator {
        sess_manager: sess_manager.clone(),
    };
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

    // Torch
    let torch_svc = BastionLabTorch::new(sess_manager.clone());
    let builder = {
        use bastionlab_torch::torch_proto::torch_service_server::TorchServiceServer;
        builder.add_service(TorchServiceServer::with_interceptor(
            torch_svc.clone(),
            token_validator.clone(),
        ))
    };

    // Polars
    let polars_svc = BastionLabPolars::new(sess_manager.clone());
    let builder = {
        use bastionlab_polars::{
            polars_proto::polars_service_server::PolarsServiceServer, BastionLabPolars,
        };
        let svc = BastionLabPolars::new(sess_manager.clone());
        match BastionLabPolars::load_dfs(&svc) {
            Ok(_) => info!("Successfully loaded saved dataframes"),
            Err(_) => info!("There was an error loading saved dataframes"),
        };
        builder.add_service(PolarsServiceServer::with_interceptor(
            polars_svc.clone(),
            token_validator.clone(),
        ))
    };

    // Linfa
    let builder = {
        use bastionlab_linfa::{
            linfa_proto::linfa_service_server::LinfaServiceServer, BastionLabLinfa,
        };
        let svc = BastionLabLinfa::new(polars_svc.clone());
        builder.add_service(LinfaServiceServer::with_interceptor(
            svc,
            token_validator.clone(),
        ))
    };

    // Conversion
    let builder = {
        use bastionlab_conversion::{
            conversion_proto::conversion_service_server::ConversionServiceServer,
            converter::Converter,
        };
        builder.add_service(ConversionServiceServer::with_interceptor(
            Converter::new(Arc::new(torch_svc.clone()), Arc::new(polars_svc.clone())),
            token_validator.clone(),
        ))
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
