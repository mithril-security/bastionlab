# Rust Server 

## Rust Server Structure

```sh
Rust Server 🚀🔐/
├── Cargo.lock
├── Cargo.toml
├── Makefile
├── tools/
│   └── config.toml
├── src/
│   └── main.rs
├── bastionlab_common/
│   ├── build.rs
│   ├── Cargo.toml
│   └── src/
│       ├── auth.rs
│       ├── config.rs
│       ├── lib.rs
│       ├── prelude.rs
│       ├── session.rs
│       └── telemetry.rs
├── bastionlab_learning/
│   ├── Cargo.toml
│   └── src/
│       ├── data/
│       │   ├── dataset.rs
│       │   ├── mod.rs
│       │   └── privacy_guard.rs
│       ├── lib.rs
│       ├── nn/
│       │   ├── mod.rs
│       │   ├── module.rs
│       │   └── parameters.rs
│       ├── optim/
│       │   ├── adam.rs
│       │   ├── mod.rs
│       │   ├── optimizer.rs
│       │   └── sgd.rs
│       ├── procedures.rs
│       └── serialization.rs
├── bastionlab_polars/
│   ├── build.rs
│   ├── Cargo.toml
│   └── src/
│       ├── access_control.rs
│       ├── composite_plan.rs
│       ├── lib.rs
│       ├── serialization.rs
│       ├── utils.rs
│       └── visitable.rs
├── bastionlab_torch/
│   ├── build.rs
│   ├── Cargo.toml
│   └── src/
│       ├── learning.rs
│       ├── lib.rs
│       ├── serialization.rs
│       ├── storage.rs
│       └── utils.rs
└── python-wheel/
    ├── create_wheel.sh
    ├── pyproject.toml
    ├── README.md
    ├── setup.py
    └── src/
        └── bastionlab_server
            ├── __init__.py
            ├── server.py
            └── version.py

```

## config.toml

```sh
tree:
├── tools/
    └── config.toml
```

### Variables {#config}

| Name                            | Default Value             |
| ------------------------------- | ------------------------- |
| client_to_enclave_untrusted_url | `"https://0.0.0.0:50056"` |
| public_keys_directory           | `"keys/"`                 |
| session_expiry_in_secs          | `1500`                    |

## main.rs

```sh
tree:
├── src/
    └── main.rs
```

### Variables {#main}

| Name                    | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| config                  | Configuration [struct](#BastionLabConfig) form the [config.toml](#config.toml) file. |
| disable_authentication  | Environmental variable `DISABLE_AUTHENTICATION`.             |
| keys                    | [`load_from_dir`](#KeyManagement) keys from [`public_keys_directory`](#Variables) |
| sess_manager            | [Session Manager](#SessionManager) smart pointer.            |
| server_cert             | X.509 Certificate PEM file created on build (`Makefile`).<br />File: `tls/host_server.pem` |
| server_key              | RSA Private Key PEM file created on build (`Makefile`).<br />File: `tls/host_server.key` |
| server_identity         | Client certificate from the private key and the X.509 cert.  |
| tee_mode, platform, uid | [Event data](#Data-Flow-setup) to [setup](#setup) telemetry. |

### Data Flow

```rust
fn main():
	config = tools/config.toml
	if authentication is enabled:
		keys = load keys from directory
	else:
		keys = None
	sess_manager = New Session Manager
	server_identity = identity from (tls/host_server.pem, tls/host_server.key)
	tee_mode, platform, uid = fill data to setup telemetry
	if bastionlab telemetry is enabled:
		telemetry::setup(platform, uid, tee_mode)
	telemetry::add_event(TelemetryEventProps started)
	
```

## bastionlab_common

```sh
tree:
├── bastionlab_common/
    ├── build.rs
    ├── Cargo.toml
    └── src/
        ├── auth.rs
        ├── config.rs
        ├── lib.rs
        ├── prelude.rs
        ├── session.rs
        └── telemetry.rs
```

### coinfig.rs

#### BastionLabConfig

*Struct* for the [`config.toml`](config.toml), with its attributes matching the variables from the config file.

##### Variables

| Name                            | Type        |
| ------------------------------- | ----------- |
| client_to_enclave_untrusted_url | `http::Uri` |
| public_keys_directory           | `String`    |
| session_expiry_in_secs          | `u64`       |

##### Methods

| Name                               | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| client_to_enclave_untrusted_socket | Returns a `Result<SocketAddr>` from the attribute: `client_to_enclave_untrusted_url`. |
| public_keys_directory              | Returns `Ok(self.public_keys_directory.clone())`.            |
| session_expiry                     | Returns a `Ok(self.session_expiry_in_secs)`.                 |

### sessions.rs

#### Session

*Struct* describing a Session.

##### Variables

| Name        | Type         |
| ----------- | ------------ |
| pubkey      | `String`     |
| user_ip     | `SocketAddr` |
| expiry      | `SystemTime` |
| client_info | `ClientInfo` |

#### SessionManager

*Struct* to manage sessions.

##### Variables

| Name           | Type                                                         |
| -------------- | ------------------------------------------------------------ |
| keys           | `Option<Mutex<KeyManagement>>`.<br />Reference: [KeyManagement](#KeyManagement) |
| sessions       | `Arc<RwLock<HashMap<[u8; 32], Session>>>`.<br />Reference [Session](#Session) |
| session_expiry | `u64`                                                        |
| challenges     | `Mutex<HashSet<[u8; 32]>>`                                   |

##### Methods

| Name            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| new             | SessionManager constructor.                                  |
| auth_enabled    | Verifies if there is anything in the attribute `keys`.       |
| verify_request  | Verifies the access token and the session validity *(existence, IP address , expiration time)* for the request. |
| get_client_info | Returns a `client_info` clone of the [session](#Session).    |
| new_challenge   |                                                              |
| check_challenge |                                                              |
| create_session  |                                                              |
| refresh_session |                                                              |

### auth.rs

#### KeyManagement

*Struct* to manage the keys of the owners and users.

##### Variables

| Name   | type                      |
| ------ | ------------------------- |
| owners | `HashMap<String, PubKey>` |
| users  | `HashMap<String, PubKey>` |

### telemetry.rs

#### Structs

##### TelemetryEvent

###### Variables

| Name        | Type                                                       |
| ----------- | ---------------------------------------------------------- |
| event_type  | `&'static str`                                             |
| props       | `TelemetryEventProps`                                      |
| time        | `SystemTime`                                               |
| client_info | `Option<ClientInfo>`. Reference: [ClientInfo](#ClientInfo) |

##### RequestEvent<'a>

###### Variables

| Name             | Type                              |
| ---------------- | --------------------------------- |
| user_id          | `&'a str`                         |
| event_type       | `&'a str`                         |
| device_id        | `&'a str`                         |
| time             | `u64`                             |
| app_version      | `&'a str`                         |
| tee_mode         | `&'a str`                         |
| user_properties  | `RequestUserProperties<'a>`       |
| event_properties | `Option<&'a TelemetryEventProps>` |

 ##### RequestUserProperties<'a>

###### Variables

| Name                      | Type              |
| ------------------------- | ----------------- |
| uptime                    | `u64`             |
| client_uid                | `Option<&'a str>` |
| client_platform_name      | `Option<&'a str>` |
| client_platform_arch      | `Option<&'a str>` |
| client_platform_version   | `Option<&'a str>` |
| client_platform_release   | `Option<&'a str>` |
| client_user_agent         | `Option<&'a str>` |
| client_user_agent_version | `Option<&'a str>` |

#### Setup

##### Variables

| Name              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| sender            | Struct `UnboundedSender<TelemetryEvent>`.<br />Reference: [TelemetryEvent](#TelemetryEvent) |
| receiver          | Struct `UnboundedReceiver<TelemetryEvent>`.<br />Reference: [TelemetryEvent](#TelemetryEvent) |
| TELEMETRY_CHANNEL | Static global variable set to the `sender` variable.         |
| first_start       | System time corresponding to “now”.                          |
| received_events   | Vector of received events.                                   |
| event             | Struct [`RequestEvent`](#RequestEvent<'a>) filled with the event data, from the parameters and the variable `properties`. |
| events            | Vector of events to send as json to the POST request.        |
| properties        | Struct [`TelemetryEvent`](#TelemetryEvent) with user properties. |
| user_properties   | Structure [`RequestUserProperties`](#RequestUserProperties<'a>) filled with the [`time`](#TelemetryEvent) and [`client_info`](#TelemetryEvent) attributes from variable `properties`. |

##### Data Flow {#setup}

```rust
fn setup(platform: String, uid: String, tee: String):
	sender, receiver = unbounded_channel <TelemetryEvent>()
	set TELEMETRY_CHANNEL sender
	firt_start = SystemTime now
	loop:
		received_events = Ver::new()
		events = Vec::new()
		while properties = receive message():
			received_events.push(properties)
		for properties in received_events:
			user_properties = RequestUserProperties { uptime : properties.time, .. }
			if properties.client_info:
				fill user_properties with properties.client_info
			app_version = env!("CARGO_PKG_VERSION")
			event = RequestEven { uid, properties.event_type, 
                					platform, properties.time, ... }
			events.push(event)
		if event is not empty:
			telemetry_url = "https://telemetry.mithrilsecurity.io/bastionlab/"
			send events as a POST request to telemetry_url
```

## [bastionlab.proto](https://github.com/mithril-security/bastionlab/blob/master/protos/bastionlab.proto)

### Messages

* **Empty** = `{}`
* **ChallengeResponse** = `{ bytes value = 1; }`
* **SessionInfo** = `{ bytes token = 1; expiry_time = 2 }`
* **ClientInfo**

#### ClientInfo

##### Variables

| Name               | Type     | Value |
| ------------------ | -------- | ----- |
| uid                | `string` | 1     |
| platform_name      | `string` | 2     |
| platform_arch      | `string` | 3     |
| platform_version   | `string` | 4     |
| platform_release   | `string` | 5     |
| user_agent         | `string` | 6     |
| user_agent_version | `string` | 7     |
| is_colab           | `bool`   | 8     |
