# Overview

## What is BastionAI?
BastionAI is confidential training technology for Deep Learning models. It addresses two important roadblocks in the wider deployment of Machine Learning (ML)-based solutions: training models with private or confidential data on an untrusted infrastructure and multi-party training where multiple data owner that do not trust each other want to collaborate to build a better model. BastionAI leverages Trusted Execution Environments (TEEs) to protect private or confidential data while it is used to train an ML model. In addition, as ML models have been demonstrated to be vulnerable to a wide swath of attacks such as model extraction, model inversion or membership inference, BastionAI optionnaly provides ways to use Differential Privacy during the whole training.

BastionAI is made of two components:
- A server running in a TEE that securely trains the model in a privacy-preserving manner.
- A Python client running on Data Owners and Data Scientists machines to connect to the server, send data and models and retrieve trained weights.

## Workflow
BastionAI seemlessly integrates with common ML practices.

![BastionAI](../../assets/workflow_bastionai.png)

Here is the typical workflow when a Data Owner wants to collaborate with an untrusted Data Scientist using BastionAI:
1. The Data Owners simply connect to the TEE through an encrypted channel and sends their private data along with a license describing what can be done with the data and who can use it.
2. The Data Scientist, if they are alowed to by the Data Owner, lists datasets available to them and finds the one sent by the Data Owner. Then, they upload their model and launch the training procedure on the selected dataset.
3. If the Data Owner allows it, they may retrieve the trained weights for further local use.
4. Optionally, they may also share the model with a BlindAI-enabled machine for deployment in production (not yet available).

Besides, the client library is fully compatible with Pytorch and in turn benefits from its interoperability with numpy, pandas and more!

## Key Features
- 🚀 An intuitive high-level API inspired by FastAI that let you focus on what matters.
- 🔒 Hardware protection for data, models, checkpoints and metrics with TEEs.
- ✅ The assurance that the intended code runs on the server (i.e. the right version of BastionAI server with the right primitives and backend).
- 🔗 Traceability in what operations run on your data or model weights.
- 🚦 A strong licensing mechanism that ensures your data or models and the models and metrics derived from them are only accesible to the intended persons.
- 🎲 Built-in Differential Privacy that applies both to the training (trained weights) and the metrics (loss, acuracy, etc.)

## What you can do with BastionAI
- Easily train state-of-the-art models from various ML subfields in a privacy-preserving manner: computer vision, computer hearing, natural language processing and more.
- Collaborate with untrusted Data Owner and Data Scientist without exposing your data or models.
- Perform computations on an untrusted environment.
- Provide guarantees to third parties, for instance, clients or regulators, that you are indeed providing data protection, through code attestation.

## What can't be achieved with BastionAI
- Deploying Deep Learning models for production use (explore [BlindAI] that has been designed for this purpose)
- Training non-Deep Learning models like sklearn (may be supported in the future).
- Performing analytics, statistics or data visualization based on data frames or databases (may be supported in the future).
- Securing an entire pipeline, including preprocessing and postprocessing steps (may be supported in the future).

[BlindAI]: https://github.com/mithril-security/blindai