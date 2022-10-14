// Copyright 2022 Mithril Security. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use http::Uri;
use serde::{de::Error, Deserialize, Deserializer};
use std::net::ToSocketAddrs;

pub mod auth;

#[derive(Deserialize, Clone, Debug)]
pub struct BastionAIConfig {
    //  Connection for Client -> Enclave communication
    #[serde(deserialize_with = "deserialize_uri")]
    pub client_to_enclave_untrusted_url: Uri,
    #[serde(default)]
    pub send_dataset_requires_auth: bool,
    #[serde(default)]
    pub send_model_requires_auth: bool,
    #[serde(default)]
    pub fetch_module_requires_auth: bool,
    #[serde(default)]
    pub fetch_dataset_requires_auth: bool,
    #[serde(default)]
    pub delete_module_requires_auth: bool,
    #[serde(default)]
    pub delete_dataset_requires_auth: bool,
    #[serde(default)]
    pub train_requires_auth: bool,
    #[serde(default)]
    pub test_requires_auth: bool,
    #[serde(default)]
    pub available_models_requires_auth: bool,
    #[serde(default)]
    pub available_datasets_requires_auth: bool,
    #[serde(default)]
    pub available_devices_requires_auth: bool,
    #[serde(default)]
    pub available_optimizers_requires_auth: bool,
    #[serde(default)]
    pub get_metric_requires_auth: bool,
}

fn uri_to_socket(uri: &Uri) -> Result<SocketAddr> {
    uri.authority()
        .context("No authority")?
        .as_str()
        .to_socket_addrs()?
        .next()
        .context("Uri could not be converted to socket")
}

impl BastionAIConfig {
    pub fn client_to_enclave_untrusted_socket(&self) -> Result<SocketAddr> {
        uri_to_socket(&self.client_to_enclave_untrusted_url)
    }
}

fn deserialize_uri<'de, D>(deserializer: D) -> Result<Uri, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    s.parse::<Uri>().map_err(D::Error::custom)
}
