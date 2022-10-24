# Overview

## What is BastionAI?
BastionAI is confidential training technology for Deep Learning models. It addresses two important roadblocks in the wider deployment of Machine Learning (ML)-based solutions: training models with private or confidential data on an untrusted infrastructure and multi-party training where multiple data owner that do not trust each other want to collaborate to build a better model. BastionAI leverages Trusted Execution Environments (TEEs) to protect private or confidential data while it is used to train an ML model. In addition, as ML models have been demonstrated to be vulnerable to a wide swath of attacks such as model extraction, model inversion or membership inference, BastionAI optionnaly provides ways to use Differential Privacy during the whole training.

BastionAI is made of two components:
- A server running in a TEE that securely trains the model in a privacy-preserving manner.
- A Python client running on Data Owners and Data Scientists machines to connect to the server, send data and models and retrieve trained weights.

## Workflow
BastionAI seemlessly integrates with common ML practices.

Here is the typical workflow when a Data Owner wants to collaborate with an untrusted Data Scientist using BastionAI:
1. The Data Owners simply connect to the TEE through an encrypted channel and sends their private data along with a license describing what can be done with the data and who can use it.
2. The Data Scientist, if they are alowed to by the Data Owner, lists datasets available to them and finds the one sent by the Data Owner.
3. They upload their model and launch the training procedure on the selected dataset.
4. If the Data Owner allows it, they may retrieve the trained weights for further local use.
5. Optionally, they may also share the model with a BlindAI-enabled machine for deployment in production.

Besides, the client library is fully compatible with Pytorch and in turn benefits from its interoperability with numpy, pandas and more!

## Key Features
- :rocket: An intuitive high-level API inspired by FastAI that let you focus on what matters.
- :factory: Hardware protection for data, models, checkpoints and metrics with TEEs.
- :heavy_check_mark: The assurance that the intended code runs on the server (i.e. the right version of BastionAI server with the right primitives and backend).
- :link: Traceability in what operations run on your data or model weights.
- :vertical_traffic_light: A strong licensing mechanism that ensures your data or models and the models and metrics derived from them are only accesible to the intended persons.
- :game_die: Built-in Differential Privacy that applies both to the training (trained weights) and the metrics (loss, acuracy, etc.)
