<p align="center">
  <img src="assets/logo.png" alt="BastionAI" width="200" height="200" />
</p>

<h1 style="font-size: 35px; font-weight: 600" align="center">Mithril Security – BastionAI</h1>

<h4 align="center">
  <a href="https://www.mithrilsecurity.io">Website</a> |
  <a href="https://bastionai.readthedocs.io">Documentation</a> |
  <a href="https://blog.mithrilsecurity.io/">Blog</a> |
  <a href="https://hub.docker.com/u/mithrilsecuritysas">Docker Hub</a> |
  <a href="https://www.linkedin.com/company/mithril-security-company">LinkedIn</a> | 
  <a href="https://www.twitter.com/mithrilsecurity">Twitter</a> | 
  <a href="https://discord.gg/TxEHagpWd4">Discord</a>
</h4>

**BastionAI** is a confidential AI training server. It introduces a new paradigm which is a merger between trusted computing and centralized learning. It provides the platform for collaborative training just like in Federated Learning (FL) settings, but with models being trained centrally in a secure, trusted computing environment. This solution makes possible for multiple parties to collectively train models without exposing either the data or the model to any of the parties. Also, Cloud providers won't be able to access either data or model in the clear thanks to the security guarantees provided by the trusted execution environment.


  ## :round_pushpin: Table of content
- [:lock: Motivation](#lock-motivation)
- [:question: Why use BastionAI](#question-why-use-bastionai)
  - [Case #1, single actor, confidential but not personal data](#case-1-single-actor-confidential-but-not-personal-data)
  - [Case #2, single actor, personal data](#case-2-single-actor-personal-data)
  - [Case #3, multiple actors, any form of confidential or personal data](#case-3-multiple-actors-any-form-of-confidential-or-personal-data)
- [:gear: Architecture](#gear-architecture)
  - [Server](#server)
  - [Client](#client)
- [:hammer_and_wrench: Key Features](#hammer_and_wrench-key-features)
  - [Remote execution](#remote-execution)
  - [Authentication](#authentication)
  - [Differential Privacy (DP)](#differential-privacy-dp)
- [:runner: Upcoming Features.](#runner-upcoming-features)
- [:wrench: How to get started](#wrench-how-to-get-started)
- [:bangbang: Disclaimer](#bangbang-disclaimer)

# :lock: Motivation
Machine learning models are often trained on large volumes of sensitive or proprietary data, and incur significant costs in development to their owners. Yet, models are often trained in the Cloud as maintenance costs of on-premise infrastructure (both hardware and skilled engineers) are prohibitive to most actors. This setting inherently poses security and privacy threats: even with end-to-end encryption, either the Cloud provider or an insider may read data as it is processed on the remote server.

Moreover, as typical Deep Learning models benefit from increased amounts of data and compute power, actors in the ML industry tend to group to share data and resources. This Federated Learning approach conflicts with security and privacy constraints, especially in the medical sector which uses very sensitive personal data.

Privacy matrix: Data privacy may be analyzed, categorized, and thought of around five principles.

| Priciple            | Objective                                                           | Chosen Technical Solution                              |
| ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------ |
| Input Privacy       | Send, store and use data/model weights securely                     | TEEs through the sealing mechanism + encrypted channel |
| Output Privacy      | Ensure output does not leak (too much) information about the inputs | Differential Privacy ([DP-SGD](https://arxiv.org/pdf/1607.00133.pdf))                          |
| Input Verification  | Ensure authentication and integrity of the inputs                   | TEEs through the attestation mechanism                 |
| Output Verification | Ensure integrity of the computation                                 | TEEs through the attestation mechanism                 |
| Flow Governance     | Provide strong access and ownership policies                        | Custom solution (outlined in the architecture section) |

# :question: Why use BastionAI
## Case #1, single actor, confidential but not personal data
In this scenario, a company wants to train an in-house model on confidential but not personal data it owns. Model and data need strong privacy guarantees when they are in the Cloud but may be used freely on-premise. Output Privacy and flow governance thus do not apply in this case. A simple TEE-based training solution is enough to cover this use case.

Potential customers: Oil & Gas companies, actually any industry that exploits confidential but not personal data and that is not willing to share them.

## Case #2, single actor, personal data
This scenario is similar to case #1 but the data at hand contains information about individuals and is thus regulated by GDPR (in Europe, but similar policies exist in other countries/states such as California and Canada). Once again flow governance is not required as there’s only one party involved but Output privacy is now mandatory. In this case, we recommend the use of TEE-based learning in conjunction with Differential Privacy.

Potential customers: organizations that exploit personal data and need not share it.

## Case #3, multiple actors, any form of confidential or personal data

As data is owned by various parties that are willing to participate in a joint training but that don’t want others to access their data, the model needs output privacy guarantees to avoid data leakage between users. This applies to disregarding whether the data is personal or not. In addition, clear policies regarding access and ownership are needed: flow governance applies in this case.

Potential customers: Hospitals and companies that use and share sensitive data about patients, more generally, any organization that takes part in Federated Learning settings.

# :gear: Architecture
<p align="center">
  <img src="assets/architecture.png" alt="BastionAI" />
</p>

## Server
The server is responsible for receiving and properly storing data and models sent by clients; also, it is responsible for securely saving models to disk, if need be (sealing). It uses an in-memory storage for all the artifacts the server receives. Both data and models are serialized as Pytorch JIT modules (TorchScript) which is the only format readable by both the client-side library (plain Pytorch) and the server-side ML back-end (libtorch) through the [tch-rs](https://github.com/LaurentMazare/tch-rs) Rust library. The server also injects differential privacy to the per-sample gradients (PSGs) during training to ensure the privacy of the output model.

## Client
We provide a lightweight client which is responsible for serializing datasets and models with Pytorch's JIT compiler before sending to the server. And to strength the security of artifacts (datasets and models,) the server returns a unique identifier for every artifact uploaded.

# :hammer_and_wrench: Key Features

## Remote execution
BastionAI uses [tch-rs](https://github.com/LaurentMazare/tch-rs) with [libtorch](https://github.com/pytorch/pytorch/tree/master/torch/csrc) (C++ bindings) as a backend for the server written in Rust. This has two benefits:
- enhanced compatibility with [BlindAI](https://github.com/mithril-security/blindai).
- reduced attack surface: less code needs to be audited, and less room for vulnerabilities and side channels. All training procedures are coded in Rust: optimizers, DP-SGD, etc.

## Authentication
BastionAI uses HMAC instead of digital signatures based on asymmetric cryptography for owner authentication (as we have no real motivation for public key infrastructure, it is better to use HMAC that are faster to make and to verify). The use of JWTs for access tokens as then may contain various structured fields that may prove useful for fine-grained access management and are inherently stateless (this feature though does not make billing easy but this matter is orthogonal).

## Differential Privacy (DP)
The DP framework we support is [DP-SGD](https://arxiv.org/pdf/1607.00133.pdf). BastionAI uniquely implements DP by using an approach of expanded weights.
The per-layer weights are expanded, which is replicating them along a new “batch” dimension as many times as the number of samples in a batch. With proper changes to the forward pass of the layer, the gradient of the expanded weights computed by the autograd engine during the backward pass are directly the PSGs. A careful analysis of the memory footprint of the approach shows that it is no more memory hungry than the hooks-based approach. In terms of compute time, we notice that the modified forward passes, at least in a naive implementation, are less efficient than the hooks-based approach or non-private learning. Nevertheless, the use of some tricks on key layers such as convolutions (e.g. grouped convolutions) allows this method to be on par with the hooks-based technique.

[Opacus](https://github.com/pytorch/opacus) is another differential privacy library but due to TorchScript's inability with backward pass hooks.

# :runner: Upcoming Features.
- Providing Attestation to BastionAI through AMD SEV.

# :wrench: How to get started
To build and run the server, first make sure you have [Pytorch's C++ libtorch library](https://pytorch.org/cppdocs/installing.html).

```shell
cd server
make LIBTORCH_PATH=/path/to/libtorch MODE=[debug|release]
make run
```

To build the client, execute the following commands, follow the instruction [here](client/README.md) 
# :bangbang: Disclaimer

BastionAI is still in alpha and is being actively developed. It is provided as is, use it at your own risk.
