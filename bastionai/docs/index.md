---
description: 'BastionAI: Fast, accessible, and privacy-friendly AI training 🚀🔒'
---

# 👋 Welcome

### What is **BastionAI?**&#x20;

**BastionAI** is a **fast, easy-to-use,** and **confidential artificial intelligence (AI) training platform**, allowing you to train your AI models on private data. BastionAI provides a new framework for multi-party computing where models and training data could be provided by different entities and yet, no private information could be leaked without the approval of any of the parties involved.

We reconcile AI and privacy by leveraging Confidential Computing for secure training. You can learn more about this technology here.

BastionAI is designed to work with secure GPU platform and a virtualized machine approach. We, therefore, plan on deploying on **AMD SEV**. More information about our **roadmap** can be found [here](https://github.com/mithril-security/bastionai/projects/1).

Our solution comes in two parts:

* A secure training solution to train _AI_ models with privacy guarantees.
* A client SDK to securely interface the trainer and provided endpoints for uploading models, uploading datasets, and many other features.

### Latest versions

* **BastionAI server:** _0.1.0_
* **BastionAI client:** _0.1.0_

### Features
#### Remote execution
BastionAI uses [tch-rs](https://github.com/LaurentMazare/tch-rs) with [libtorch](https://github.com/pytorch/pytorch/tree/master/torch/csrc) (C++ bindings) as a backend for the server written in Rust. This has two benefits:
- enhanced compatibility with [BlindAI](https://github.com/mithril-security/blindai).
- reduced attack surface: less code needs to be audited, and less room for vulnerabilities and side channels. All training procedures are coded in Rust: optimizers, DP-SGD, etc.

#### Authentication
BastionAI uses HMAC instead of digital signatures based on asymmetric cryptography for owner authentication (as we have no real motivation for public key infrastructure, it is better to use HMAC that are faster to make and to verify). The use of JWTs for access tokens as then may contain various structured fields that may prove useful for fine-grained access management and are inherently stateless (this feature though does not make billing easy but this matter is orthogonal).

#### Differential Privacy (DP)
The DP framework we support is [DP-SGD](https://arxiv.org/pdf/1607.00133.pdf). BastionAI uniquely implements DP by using an approach of expanded weights.
The per-layer weights are expanded, which is replicating them along a new “batch” dimension as many times as the number of samples in a batch. With proper changes to the forward pass of the layer, the gradient of the expanded weights computed by the autograd engine during the backward pass are directly the PSGs. A careful analysis of the memory footprint of the approach shows that it is no more memory hungry than the hooks-based approach. In terms of compute time, we notice that the modified forward passes, at least in a naive implementation, are less efficient than the hooks-based approach or non-private learning. Nevertheless, the use of some tricks on key layers such as convolutions (e.g. grouped convolutions) allows this method to be on par with the hooks-based technique.

### Who made BastionAI?&#x20;

BastionAI was developed by **Mithril Security**. **Mithril Security** is a startup focused on confidential machine learning based on **Intel SGX** and **AMD SEV** technology. We provide an **open-source AI inference solution**, **open-source AI training solution**, **allowing easy and fast deployment of neural networks, with strong security properties** provided by confidential computing by performing the computation in a hardware-based **Trusted Execution Environment** (_TEE_) or simply **enclaves**.




