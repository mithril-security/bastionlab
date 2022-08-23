<p align="center">
  <img src="assets/logo.png" alt="BastionAI" width="200" height="200" />
</p>

<h1 align="center">Mithril Security – BastionAI</h1>

<h4 align="center">
  <a href="https://www.mithrilsecurity.io">Website</a> |
  <a href="https://blog.mithrilsecurity.io/">Blog</a> |
  <a href="https://www.linkedin.com/company/mithril-security-company">LinkedIn</a> | 
  <a href="https://www.twitter.com/mithrilsecurity">Twitter</a> | 
  <a href="https://discord.gg/TxEHagpWd4">Discord</a>
</h4>

<h3 align="center">Fast, accessible and privacy friendly AI training 🚀🔒</h3>

BastionAI is a confidential deep learning framework. BastionAI enables data scientists to train their models on sensitive data, while ensuring to the data providers that no third party will have access to their data.

With BastionAI, you can:
- Have users send data for finetuning a model, for instance through the Cloud, without data being exposed in clear. 
- Securely train a model on datasets from multiple data owners, without having any party have their data being shown to third parties.

BastionAI uses state-of-the art Privacy Enhancing Technologies, such as [Confidential Computing](https://blog.mithrilsecurity.io/confidential-computing-explained-part-1-introduction/) and [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy) (DP), to protect data from being accessed by outside parties. The server side is implemented using [Rust]((https://www.rust-lang.org/)), for memory safety. 🦀🦀

BastionAI currently supports the confidential training of [PyTorch](https://pytorch.org/) models, with DP implemented through [Opacus](https://github.com/pytorch/opacus). 

## :lock: Motivation

Today, most AI tools offer no privacy by design mechanisms. This means that when multiple parties want to pool data for training an AI model, for instance in healthcare, very sensitive data can be exposed to multiple third parties, posing security and privacy issues.

For instance, if we imagine hospitals wanting 

We illustrate it below with the use of AI for voice assistants. Audio recordings are often sent to the Cloud to be analysed, leaving conversations exposed to leaks and uncontrolled usage without users’ knowledge or consent.

Currently, even though data can be sent securely with TLS, some stakeholders in the loop can see and expose data : the AI company renting the machine, the Cloud provider or a malicious insider. 

## Quick tour

BastionAI is a confidential training framework, built on top of PyTorch. 
To train a model on confidential data, 

### Finetune BERT on confidential data

We provide an example.

## :white_check_mark: Key Features

- **Confidentiality made easy**: Easily train state-of-the-art models with confidentiality thanks to the use of secure enclaves and Differential Privacy.
- **Transparency**: Provide guarantees to third parties, for instance clients or regulators, that you are indeed providing **data protection**, through **code attestation**.
- **Extensible usage**: Explore different scenarios from confidential multi party training of cancer detection AI from a ResNet, to the fine-tuning of GPT models on confidential text, such as emails or documents.

# :bangbang: Disclaimer

BastionAI is still in alpha and is being actively developed. It is provided as is, use it at your own risk.
