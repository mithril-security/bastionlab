# Rust Server 

## Rust Server Structure

<pre>
<b>Rust Server 🚀🔐/</b>
├── <i>Cargo.lock</i>
├── <i>Cargo.toml</i>
├── Makefile
├── <b>tools/</b>
│   └── <a href="#config.toml">config.toml</a>
├── <b>src/</b>
│   └── <a href="#main.rs">main.rs</a>
├── <a href="#bastionlab_common">bastionlab_common/</a>
│   ├── <i>build.rs</i>
│   ├── <i>Cargo.toml</i>
│   └── <b>src/</b>
│       ├── <a href="#lib.rs-common">lib.rs</a>
│       ├── <a href="#prelude.rs-common">prelude.rs</a>
│       ├── <a href="#config.rs">config.rs</a>
│       ├── <a href="#session.rs">session.rs</a>
│       ├── <a href="#auth.rs">auth.rs</a>
│       └── <a href="#telemetry.rs">telemetry.rs</a>
├── <a href="#bastionlab_polars">bastionlab_polars/</a>
│   ├── <i>build.rs</i>
│   ├── <i>Cargo.toml</i>
│   └── <b>src/</b>
│       ├── access_control.rs
│       ├── composite_plan.rs
│       ├── lib.rs
│       ├── serialization.rs
│       ├── utils.rs
│       └── visitable.rs
├── <a href="#bastionlab_torch">bastionlab_torch/</a>
│   ├── <i>build.rs</i>
│   ├── <i>Cargo.toml</i>
│   └── <b>src/</b>
│       ├── learning.rs
│       ├── lib.rs
│       ├── serialization.rs
│       ├── storage.rs
│       └── utils.rs
└── <a href="#bastionlab_learning">bastionlab_learning/</a>
    ├── <i>Cargo.toml</i>
    └── <b>src/</b>
        ├── <b>data/</b>
        │   ├── dataset.rs
        │   ├── mod.rs
        │   └── privacy_guard.rs
        ├── lib.rs
        ├── <b>nn/</b>
        │   ├── mod.rs
        │   ├── module.rs
        │   └── parameters.rs
        ├── <b>optim/</b>
        │   ├── adam.rs
        │   ├── mod.rs
        │   ├── optimizer.rs
        │   └── sgd.rs
        ├── procedures.rs
        └── serialization.rs
</pre>

## config.toml

```sh
tree:
├── tools/
    └── config.toml
```

This configuration file is translated into a [*struct*](#BastionLabConfig) used in the [main](#main.rs).

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

| Name                           | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| config                         | Configuration [struct](#BastionLabConfig) form the [config.toml](#config.toml) file. |
| disable_authentication         | Environmental variable `DISABLE_AUTHENTICATION`.             |
| keys                           | [`load_from_dir`](#KeyManagement) keys from [`public_keys_directory`](#Variables) |
| sess_manager                   | [Session Manager](#SessionManager) smart pointer.            |
| server_cert                    | X.509 Certificate PEM file created on build (`Makefile`).<br />File: `tls/host_server.pem` |
| server_key                     | RSA Private Key PEM file created on build (`Makefile`).<br />File: `tls/host_server.key` |
| server_identity                | Client certificate from the private key and the X.509 cert.  |
| tee_mode, platform, uid        | [Event data](#Data-Flow-setup) to [setup](#setup) telemetry. |
| `TelemetryEventProps::Started` | [`Started`](#variants-TelemetryEventProps) variant from `TelemetryEventProps` |
| builder                        | Builder to configure a [`Server`](https://docs.rs/tonic/latest/tonic/transport/struct.Server.html). |
| svc                            | Service for the creation of a router with [add_service](https://docs.rs/tonic/latest/tonic/transport/struct.Server.html#method.add_service). |
| addr                           | [`client_to_enclave_untrusted_socket`](#Methods-BastionLabConfig). |

### Data Flow {#main}

```rust
fn main():
	config = tools/config.toml
	if authentication is enabled:
		keys = load keys from directory
	else:
		keys = None
	sess_manager = New Session Manager
	server_identity = identity from ("tls/host_server.pem", "tls/host_server.key")
	tee_mode, platform, uid = fill data to setup telemetry
	if bastionlab telemetry is enabled:
		telemetry::setup(platform, uid, tee_mode)
	telemetry::add_event(TelemetryEventProps::Started)
	builder = setting up TLS with server_identity
	// Session
	builder = {
        svc = new SessionGrpcService (sess_manager)
        // SessionServiceServer
        builder.add_service (Session service server (svc))
	}
	// Polars
	builder = {
    	svc = new BastionLabPolars (sess_manager)
        BastionLabPolars::load dataframes (svc)
        // PolarsServiceServer
        builder.add_service (Polars service server (svc))
	}

	// Torch
	builder = {
   		svc = new BastionLabTorch (sess_manager))
		// TorchServiceServer
		builder.add_service(Torch service server (svc))
	}

	addr = socket
	// Serve
	builder.serve(addr)
	
```

* [SessionServiceServer](https://github.com/mithril-security/bastionlab/blob/master/server/bastionlab_common/build.rs)
  * Compiled from proto: [bastionlab.proto](protos.md#bastionlab.proto)
* [PolarsServiceServer](https://github.com/mithril-security/bastionlab/blob/master/server/bastionlab_polars/build.rs)
  * Compiled from proto: [bastionlab_polars.proto](protos.md#bastionlab_polars.proto)
* [TorchServiceServer](https://github.com/mithril-security/bastionlab/blob/master/server/bastionlab_torch/build.rs)
  * Compiled from proto: [bastionlab_torch.proto](protos.md#bastionlab_torch.proto)

## bastionlab_common

<pre>
<b>tree:</b>
├── <b>bastionlab_common/</b>
    ├── <i>build.rs</i>				<span style="color:gray"># compile ../../protos/bastionlab.proto</span>
    ├── <i>Cargo.toml</i>				<span style="color:gray"># Manifest</span>
    └── <b>src/</b>
        ├── <a href="#lib.rs-common">lib.rs</a>
        ├── <a href="#prelude.rs-common">prelude.rs</a>
        ├── <a href="#config.rs">config.rs</a>
        ├── <a href="#session.rs">session.rs</a>
        ├── <a href="#auth.rs">auth.rs</a>
        └── <a href="#telemetry.rs">telemetry.rs</a>
</pre>

### lib.rs {#common}

Includes *`bastionlab`* proto items.

### prelude.rs {#common}

#### Imports

| anyhow  | log   | std                          |
| ------- | ----- | ---------------------------- |
| anyhow  | debug | `collections::HashMap`       |
| bail    | error | `hash::{Hash, Hasher}`       |
| ensure  | info  | `sync::{Arc, Mutex, RwLock}` |
| Context | trace |                              |
| Result  | warn  |                              |

### config.rs

#### BastionLabConfig

*Struct* for the [`config.toml`](config.toml), with its attributes matching the variables from the config file.

##### Variables

| Name                            | Type        |
| ------------------------------- | ----------- |
| client_to_enclave_untrusted_url | `http::Uri` |
| public_keys_directory           | `String`    |
| session_expiry_in_secs          | `u64`       |

##### Methods {#BastionLabConfig}

| Name                               | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| client_to_enclave_untrusted_socket | Returns a `Result<SocketAddr>` from the attribute: `client_to_enclave_untrusted_url`. |
| public_keys_directory              | Returns `Ok(self.public_keys_directory.clone())`.            |
| session_expiry                     | Returns a `Ok(self.session_expiry_in_secs)`.                 |

### session.rs

#### Session

*Struct* describing a Session.

##### Variables {#session}

| Name        | Type         |
| ----------- | ------------ |
| pubkey      | `String`     |
| user_ip     | `SocketAddr` |
| expiry      | `SystemTime` |
| client_info | `ClientInfo` |

#### SessionManager

*Struct* to manage sessions.

##### Variables {#sessionManager}

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
| get_client_info | Returns a Clint_info` clone of the [session](#Session).      |
| new_challenge   | Creates challenge to verify the request.                     |
| check_challenge | Checks the request's challenge.                              |
| create_session  | Creates and inserts a new session when auth is enabled or disabled. |

###### create_session Data Flow

```rust
fn when_auth_enabled(request, sessions):
	challenge = check_challenge(request)
	for key in request.metadata().keys():
		match key {
			KeyRef::Binary(key) => {
        	    key = key.to_string().strip_suffix("-bin")
        	    if "signature-" in key:
            		...
			}
		}

fn when_auth_disabled(sessions):
	token = [0u8; 32]
	expiry = now
	sessions.insert(token, session {pubkey: 'unauthenticated', expiry, ..} )
	return (sessionInfo { token, expiry_time: session_expiry * 1000})

fn create_session(request: Request<ClientInfo>):
	if auth is enabled:
		when_auth_enabled(request, sessions)
	else:
		when_auth_disabled(sessions)

```

#### SessionGrpcService

`Struct` to manage **gRPC**.

##### Variable

| Name         | Type                  |
| ------------ | --------------------- |
| sess_manager | `Arc<SessionManager>` |

##### Methods

| Name | Description                                                  |
| ---- | ------------------------------------------------------------ |
| new  | Constructor for `SessionGrpcService` filling `sess_manager`. |

### auth.rs

#### KeyManagement

*Struct* to manage the keys of the owners and users.

##### Variables {#keyManagement}

| Name   | type                      |
| ------ | ------------------------- |
| owners | `HashMap<String, PubKey>` |
| users  | `HashMap<String, PubKey>` |

### telemetry.rs

#### Variables {#telemetry}

A static global variable for the telemetry channel's sender.

| Name              | type                                        |
| ----------------- | ------------------------------------------- |
| TELEMETRY_CHANNEL | `OnceCell<UnboundedSender<TelemetryEvent>>` |

#### Functions {#telemetry}

| Name      | Description                                     |
| --------- | ----------------------------------------------- |
| add_event | Sends an event through the `TELEMETRY_CHANNEL`. |

#### Enum

##### TelemetryEventProps

###### Variants {#TelemetryEventProps}

| Name                   | Variables        | Type             |
| ---------------------- | ---------------- | ---------------- |
| **Started**            |                  |                  |
| **RunQuery**           | dataset_name     | `Option<String>` |
|                        | dataset_hash     | `Option<String>` |
|                        | time_taken       | `f64`            |
| **SendDataFrame**      | dataset_name     | `Option<String>` |
|                        | dataset_hash     | `Option<String>` |
|                        | time_taken       | `f64`            |
| **FetchDataFrame**     | dataset_name     | `Option<String>` |
|                        | request_accepted | `bool`           |
| **ListDataFrame**      |                  |                  |
| **GetDataFrameHeader** | dataset_name     | `Option<String>` |
| **SaveDataFrame**      | dataset_name     | `Option<String>` |
| **SendModel**          | model_name       | `Option<String>` |
|                        | model_hash       | `Option<String>` |
|                        | model_size       | `usize`          |
|                        | time_taken       | `f64`            |
| **SendDataset**        | dataset_name     | `Option<String>` |
|                        | dataset_hash     | `Option<String>` |
|                        | dataset_size     | `usize`          |
|                        | time_taken       | `f64`            |
| **TrainerLog**         | log_type         | `Option<String>` |
|                        | model_hash       | `Option<String>` |
|                        | dataset_hash     | `Option<String>` |
|                        | time             | `u128`           |

###### Functions

| Name       | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| event_type | Return [`TelemetryEventProps`](#Variables-TelemetryEventProps) event name. |

#### Structs

##### TelemetryEvent

###### Variables {#telemetryEvent}

| Name        | Type                                                         |
| ----------- | ------------------------------------------------------------ |
| event_type  | `&'static str`                                               |
| props       | `TelemetryEventProps`                                        |
| time        | `SystemTime`                                                 |
| client_info | `Option<ClientInfo>`. Reference: [ClientInfo](protos.md#ClientInfo) |

##### RequestEvent<'a>

###### Variables {#requestEvent}

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

###### Variables {#requestUserProperties}

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

##### Variables {#setup}

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

## bastionlab_polars

<pre>
<b>tree:</b>
├── <b>bastionlab_polars/</b>
    ├── <i>build.rs</i>				<span style="color:gray"># compile ../../protos/bastionlab.proto</span>
    ├── <i>Cargo.toml</i>				<span style="color:gray"># Manifest</span>
    └── <b>src/</b>
        ├── access_control.rs
        ├── composite_plan.rs
        ├── lib.rs
        ├── serialization.rs
        ├── utils.rs
        └── visitable.rs
</pre>



## bastionlab_torch

<pre>
<b>tree:</b>
├── <b>bastionlab_torch/</b>
    ├── <i>build.rs</i>				<span style="color:gray"># compile ../../protos/bastionlab.proto</span>
    ├── <i>Cargo.toml</i>				<span style="color:gray"># Manifest</span>
    └── <b>src/</b>
        ├── learning.rs
        ├── lib.rs
        ├── serialization.rs
        ├── storage.rs
        └── utils.rs
</pre>



## bastionlab_learning

<pre>
<b>tree:</b>
└── <b>bastionlab_learning/</b>
    ├── <i>Cargo.toml</i>				<span style="color:gray"># Manifest</span>
    └── <b>src/</b>
        ├── <b>data/</b>
        │   ├── dataset.rs
        │   ├── mod.rs
        │   └── privacy_guard.rs
        ├── lib.rs
        ├── <b>nn/</b>
        │   ├── mod.rs
        │   ├── module.rs
        │   └── parameters.rs
        ├── <b>optim/</b>
        │   ├── adam.rs
        │   ├── mod.rs
        │   ├── optimizer.rs
        │   └── sgd.rs
        ├── procedures.rs
        └── serialization.rs
</pre>

