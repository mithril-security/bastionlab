use env_logger::Env;
use log::error;
use log::info;
use tokio::runtime::Builder;
use tokio::signal;

use pyo3::{prelude::*, wrap_pyfunction};

use http::Uri;

use bastionlab_common::authentication::*;
use bastionlab_common::config::*;
use rcgen::generate_simple_self_signed;
use tonic::transport::Identity;

/// Start BastionLab server.
/// Args:
///     port (Optional[int]): The port on which the server will listen to connections. Default to 50056.
///     session_expiration (Optional[int]): The amount of seconds before a session expire. Default to 150.
///     keys_path (Optional[str]): The path where the owners and users keys can be found. Leaving this to none will disable the authentification. Default to None.
#[pyfunction]
#[pyo3(text_signature = "(port=50056, session_expiration=150, keys_path=None, /)")]
fn start(
    _py: Python,
    port: Option<i32>,
    session_expiration: Option<i32>,
    keys_path: Option<&str>,
) -> PyResult<()> {
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let mut port_target: i32 = 50056;
    let mut session_time: u64 = 150;

    if let Some(port_nb) = port {
        port_target = port_nb;
    };

    if let Some(session) = session_expiration {
        session_time = session as u64;
    };

    let config = BastionLabConfig {
        client_to_enclave_untrusted_url: format!("https://0.0.0.0:{port_target}")
            .parse::<Uri>()
            .unwrap(),
        public_keys_directory: "Keys".to_string(),
        session_expiry_in_secs: session_time,
    };

    let mut keys = None;

    if let Some(keys_dir) = keys_path {
        let keys_mgr_res = KeyManagement::load_from_dir(keys_dir.to_string());
        match keys_mgr_res {
            Ok(keys_mgr) => {
                keys = Some(keys_mgr);
                info!("Authentification enabled");
            }
            Err(err) => error!("Error while loading users keys. Reason: {:?}", err),
        }
    };

    if keys.is_none() {
        info!("Authentification disabled");
    }

    //TODO: To be replaced when TEE support is implemented
    let subject_alt_names = vec!["bastionlab-server".to_string()];
    let cert = generate_simple_self_signed(subject_alt_names).unwrap();
    let server_identity = Identity::from_pem(
        cert.serialize_pem().unwrap(),
        cert.serialize_private_key_pem(),
    );

    let _thread = rt.spawn(async move {
        bastionlab_common::start(config, keys, server_identity)
            .await
            .expect("Server error");
    });

    // Unfortunately required, otherwise the server won't shutdown with Ctrl+C...
    rt.block_on(async move {
        signal::ctrl_c().await.expect("Can't listen to SIGINT");
    });

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn bastionlab_server(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    m.add_function(wrap_pyfunction!(start, m)?)?;
    Ok(())
}
