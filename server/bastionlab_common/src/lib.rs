pub mod auth;
pub mod config;
pub mod prelude;
pub mod session;
pub mod telemetry;

pub mod session_proto {
    tonic::include_proto!("bastionlab");
}
