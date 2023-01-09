use crate::prelude::*;
use crate::session_proto::ClientInfo;
use once_cell::sync::OnceCell;
use serde::Serialize;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{self, UnboundedSender};

static TELEMETRY_CHANNEL: OnceCell<UnboundedSender<TelemetryEvent>> = OnceCell::new();

#[derive(Debug, Clone, Serialize)]
pub enum TelemetryEventProps {
    Started {},
    RunQuery {
        dataset_name: Option<String>,
        dataset_hash: Option<String>,
        time_taken: f64,
    },
    SendDataFrame {
        dataset_name: Option<String>,
        dataset_hash: Option<String>,
        time_taken: f64,
    },
    FetchDataFrame {
        dataset_name: Option<String>,
        request_accepted: bool,
    },
    ListDataFrame {},
    GetDataFrameHeader {
        dataset_name: Option<String>,
    },
    SaveDataframe {
        dataset_name: Option<String>,
    },
    DeleteDataframe {
        dataset_name: Option<String>,
    },
    // Torch
    SendModel {
        model_name: Option<String>,
        model_hash: Option<String>,
        model_size: usize,
        time_taken: f64,
    },
    SendDataset {
        dataset_name: Option<String>,
        dataset_hash: Option<String>,
        dataset_size: usize,
        time_taken: f64,
    },
    TrainerLog {
        log_type: Option<String>,
        model_hash: Option<String>,
        dataset_hash: Option<String>,
        time: u128,
    },
}

#[derive(Debug, Clone)]
pub struct TelemetryEvent {
    event_type: &'static str,
    props: TelemetryEventProps,
    time: SystemTime,
    client_info: Option<ClientInfo>,
}

impl TelemetryEventProps {
    fn event_type(&self) -> &'static str {
        match self {
            TelemetryEventProps::Started { .. } => "started",
            TelemetryEventProps::RunQuery { .. } => "run_query",
            TelemetryEventProps::FetchDataFrame { .. } => "fetch_data_frame",
            TelemetryEventProps::SendDataFrame { .. } => "send_data_frame",
            TelemetryEventProps::ListDataFrame { .. } => "list_data_frame",
            TelemetryEventProps::GetDataFrameHeader { .. } => "get_data_frame_header",
            TelemetryEventProps::SaveDataframe { .. } => "save_data_frame",
            TelemetryEventProps::DeleteDataframe { .. } => "delete_data_frame",
            // torch
            TelemetryEventProps::SendModel { .. } => "send_model",
            TelemetryEventProps::SendDataset { .. } => "send_dataset",
            TelemetryEventProps::TrainerLog { .. } => "trainer_log",
        }
    }
}

pub fn add_event(event: TelemetryEventProps, client_info: Option<ClientInfo>) {
    if let Some(sender) = TELEMETRY_CHANNEL.get() {
        let _ = sender.send(TelemetryEvent {
            event_type: event.event_type(),
            props: event,
            time: SystemTime::now(),
            client_info,
        });
    }
}

#[derive(Debug, Serialize)]
struct RequestEvent<'a> {
    user_id: &'a str,
    event_type: &'a str,
    device_id: &'a str,
    time: u64,
    app_version: &'a str,
    tee_mode: &'a str,
    user_properties: RequestUserProperties<'a>,
    event_properties: Option<&'a TelemetryEventProps>,
}

#[derive(Debug, Serialize, Default)]
struct RequestUserProperties<'a> {
    uptime: u64,
    client_uid: Option<&'a str>,
    client_platform_name: Option<&'a str>,
    client_platform_arch: Option<&'a str>,
    client_platform_version: Option<&'a str>,
    client_platform_release: Option<&'a str>,
    client_user_agent: Option<&'a str>,
    client_user_agent_version: Option<&'a str>,
    is_colab: bool,
}

pub fn setup(platform: String, uid: String, tee: String) -> Result<()> {
    let (sender, mut receiver) = mpsc::unbounded_channel::<TelemetryEvent>();

    TELEMETRY_CHANNEL.set(sender).unwrap();

    let first_start = SystemTime::now();
    tokio::task::spawn(async move {
        loop {
            {
                let mut received_events = Vec::new();
                let mut events = Vec::new();
                while let Ok(properties) = receiver.try_recv() {
                    received_events.push(properties);
                }

                for properties in &received_events {
                    let mut user_properties = RequestUserProperties {
                        uptime: properties
                            .time
                            .duration_since(first_start)
                            .unwrap()
                            .as_secs(),
                        ..Default::default()
                    };

                    if let Some(ref client_info) = properties.client_info {
                        user_properties.client_uid = Some(client_info.uid.as_ref());
                        user_properties.client_platform_name =
                            Some(client_info.platform_name.as_ref());
                        user_properties.client_platform_arch =
                            Some(client_info.platform_arch.as_ref());
                        user_properties.client_platform_version =
                            Some(client_info.platform_version.as_ref());
                        user_properties.client_platform_release =
                            Some(client_info.platform_release.as_ref());
                        user_properties.client_user_agent = Some(client_info.user_agent.as_ref());
                        user_properties.client_user_agent_version =
                            Some(client_info.user_agent_version.as_ref());
                        user_properties.is_colab = client_info.is_colab;
                    }

                    let event_type = properties.event_type;
                    let (user_id, device_id, app_version) =
                        (uid.as_ref(), platform.as_ref(), env!("CARGO_PKG_VERSION"));

                    let event = RequestEvent {
                        user_id,
                        event_type,
                        device_id,
                        time: properties
                            .time
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        app_version,
                        tee_mode: &tee,
                        user_properties,
                        event_properties: Some(&properties.props),
                    };

                    events.push(event);
                }

                if !events.is_empty() {
                    let response = reqwest::Client::new()
                        .post("https://telemetry.mithrilsecurity.io/bastionlab/")
                        .timeout(Duration::from_secs(60))
                        .json(&events)
                        .send()
                        .await;
                    if let Err(e) = response {
                        debug!("Cannot contact telemetry server: {}", e);
                    }
                    debug!("Telemetry OK");
                };
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });

    Ok(())
}
