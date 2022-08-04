<p align="center">
  <img src="assets/logo.png" alt="BastionAI" width="200" height="200" />
</p>
<h1 style="font-size: 35px; font-weight: 600" align="center">Mithril Security – BastionAI</h1>

**BastionAI** is a confidential AI training server. It introduces a new paradigm which is a merger between trusted computing and centralized learning. It provides the platform for collaborative training just like in Federated Learning (FL) settings, but with models being trained centrally in a secure, trusted computing environment. This solution makes possible for multiple parties to collectively train models without exposing either the data or the model to any of the parties. Also, Cloud providers won't be able to access either data or model in the clear thanks to the security guarantees provided by the trusted execution environment.

# Architecture
## Server
The server is responsible for receiving and properly storing data and models sent by clients; also, it is responsible for securely saving models to disk, if need be (sealing). It uses an in-memory storage for all the artifacts the server receives. Both data and models are serialized as Pytorch JIT modules (TorchScript) which is the only format readable by both the client-side library (plain Pytorch) and the server-side ML back-end (libtorch) through the [tch-rs](https://github.com/LaurentMazare/tch-rs) Rust library. The server also injects differential privacy to the per-sample gradients (PSGs) during training to ensure the privacy of the output model.

## Client
We provide a lightweight client which is responsible for serializing data and models with Pytorch's JIT compiler before sending to the server. And to strength the security of artifacts (datasets and models,) the server returns a unique identifier for every artifact uploaded.

  ## :round_pushpin: Table of content
- [:lock: Motivation](#lock-motivation)

## :lock: Motivation
Machine learning models are often trained on large volumes of sensitive or proprietary data, and incur significant costs in development to their owners. Yet, models are often trained in the Cloud as maintenance costs of on-premise infrastructures (both hardware and skilled engineers) are prohibitive to most actors. This setting inherently poses security and privacy threats: even with end-to-end encryption, either the Cloud provider or an insider may read data as it is processed on the remote server.

Moreover, as typical Deep Learning models benefit from increased amounts of data and compute power, actors in the ML industry tend to group to share data and resources. This Federated Learning approach conflicts with security and privacy constraints, especially in the medical sector which uses very sensitive personal data.

Privacy matrix: Data privacy may be analyzed, categorized, and thought of around five principles.

| Priciple | Objective | Chosen Technical Solution |
|----------| ----------| --------------------------|
| Input Privacy | Send, store and use data/model weights securely | TEEs through the sealing mechanism + encrypted channel |
| Output Privacy |Ensure output does not leak (too much) information about the inputs|Differential Privacy (DP-SGD)|
| Input Verification |Ensure authentication and integrity of the inputs|TEEs through the attestation mechanism|
| Output Verification |Ensure integrity of the computation|TEEs through the attestation mechanism|
| Flow Governance |Provide strong access and ownership policies|Custom solution (outlined in the architecture section)|
