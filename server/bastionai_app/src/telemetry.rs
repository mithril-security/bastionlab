use once_cell::sync::OnceCell;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::remote_torch::ClientInfo;

use log::debug;
use serde::Serialize;
use tokio::sync::mpsc::{self, UnboundedSender};

static TELEMETRY_CHANNEL: OnceCell<UnboundedSender<TelemetryEvent>> = OnceCell::new();
static TELEMETRY_SERVER_ADDR: &'static str = "https://telemetry.mithrilsecurity.io/bastionai";
#[derive(Debug, Clone, Serialize)]
pub enum TelemetryEventProps {
    Started {},
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
    is_client_event: bool,
    props: TelemetryEventProps,
    time: SystemTime,
    client_info: Option<ClientInfo>,
}

impl TelemetryEventProps {
    fn event_type(&self) -> &'static str {
        match self {
            TelemetryEventProps::Started { .. } => "started",
            TelemetryEventProps::SendModel { .. } => "send_model",
            TelemetryEventProps::SendDataset { .. } => "send_dataset",
            TelemetryEventProps::TrainerLog { .. } => "trainer_log",
        }
    }
    fn user_event_type(&self) -> Option<&'static str> {
        match self {
            TelemetryEventProps::Started { .. } => None,
            TelemetryEventProps::SendModel { .. } => Some("client_send_model"),
            TelemetryEventProps::SendDataset { .. } => Some("client_send_dataset"),
            TelemetryEventProps::TrainerLog { .. } => None,
        }
    }
}

pub fn add_event(event: TelemetryEventProps, client_info: Option<ClientInfo>) {
    if let Some(sender) = TELEMETRY_CHANNEL.get() {
        if client_info.is_some() && event.user_event_type().is_some() {
            // send the event as a user event too
            let _ = sender.send(TelemetryEvent {
                is_client_event: true,
                event_type: event.user_event_type().unwrap(),
                props: event.clone(),
                time: SystemTime::now(),
                client_info: client_info.clone(),
            });
        }
        let _ = sender.send(TelemetryEvent {
            is_client_event: false,
            event_type: event.event_type(),
            props: event,
            time: SystemTime::now(),
            client_info,
        });
    }
    // else, telemetry is disabled
}

#[derive(Debug, Serialize)]
struct RequestEvent<'a> {
    user_id: &'a str,
    event_type: &'a str,
    device_id: &'a str,
    time: u64,
    app_version: &'a str,
    user_properties: ReqestUserProperties<'a>,
    event_properties: Option<&'a TelemetryEventProps>,
}

#[derive(Debug, Serialize, Default)]
struct ReqestUserProperties<'a> {
    uptime: u64,
    client_uid: Option<&'a str>,
    client_platform_name: Option<&'a str>,
    client_platform_arch: Option<&'a str>,
    client_platform_version: Option<&'a str>,
    client_platform_release: Option<&'a str>,
    client_user_agent: Option<&'a str>,
    client_user_agent_version: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct AmplitudeRequest<'a> {
    api_key: &'a str,
    events: &'a Vec<RequestEvent<'a>>,
}

pub fn setup(platform: String, uid: String) -> anyhow::Result<()> {
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
                    let mut user_properties = ReqestUserProperties {
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
                    }

                    let event_type = properties.event_type;
                    let (user_id, device_id, app_version) = {
                        let client_info = properties.client_info.as_ref();
                        if properties.is_client_event && client_info.is_some() {
                            // this is a client event
                            let ua = client_info.unwrap();
                            (
                                ua.uid.as_ref(),
                                ua.user_agent.as_ref(),
                                ua.user_agent_version.as_ref(),
                            )
                        } else {
                            // this is a server event
                            (uid.as_ref(), platform.as_ref(), env!("CARGO_PKG_VERSION"))
                        }
                    };

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
                        user_properties,
                        event_properties: Some(&properties.props),
                    };

                    events.push(event);
                }

                //We send using the server, the differents event in the db
                if !events.is_empty() {
                    let response = reqwest::Client::new()
                        .post(TELEMETRY_SERVER_ADDR)
                        .timeout(Duration::from_secs(60))
                        .json(&events)
                        .send()
                        .await;
                    if let Err(e) = response {
                        debug!("Cannot contact telemetry server: {}", e);
                    }
                };
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    Ok(())
}
