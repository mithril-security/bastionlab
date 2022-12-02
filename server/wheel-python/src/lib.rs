use tokio::runtime::Builder;
use tokio::signal;

use pyo3::{wrap_pyfunction, prelude::*};

use http::Uri;

use bastionlab_common::config::*;
use bastionlab_common::authentication::*;
use tonic::{
    transport::Identity,
};
use rcgen::generate_simple_self_signed;

#[pyfunction]
fn start(_py: Python, port: Option<i32>, session_expiration: Option<i32>)-> PyResult<bool> {
    
    let rt = Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut port_target: i32 = 50056;
    let mut session_time: u64 = 150;

    if let Some(port_nb) = port {
        port_target = port_nb;
    };

    if let Some(session) = session_expiration {
        session_time = session as u64;
    };

    let config = BastionLabConfig {
        client_to_enclave_untrusted_url: format!("https://0.0.0.0:{port_target}").parse::<Uri>().unwrap(),
        public_keys_directory: "Keys".to_string(),
        session_expiry_in_secs: session_time,
    };

    let keys = KeyManagement::default();

    //TODO: To be replaced when TEE support is implemented 
    let subject_alt_names = vec!["bastionlab-server".to_string()];
    let cert = generate_simple_self_signed(subject_alt_names).unwrap();
    let server_identity = Identity::from_pem(cert.serialize_pem().unwrap(), cert.serialize_private_key_pem());

    rt.enter();

    rt.spawn(async move {
        bastionlab_common::start(config, keys, server_identity).await;
    });

    // Unfortunately required, otherwise the server won't shutdown with Ctrl+C...
    rt.block_on(async move {
        signal::ctrl_c().await;
    });

    Ok(true)
}

/// A Python module implemented in Rust.
#[pymodule]
fn bastionlab_server(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    Ok(())
}