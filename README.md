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

Our solution leverages Confidential VMs technology. We will first cover AMD SEV SNP, and we will support Intel TDX and Nvidia Confidential Computing as those technologies mature and become available.

## 🔒 Motivation

Today, most AI tools offer no privacy by design mechanisms. This means that when multiple parties want to pool data for training an AI model, for instance in healthcare, very sensitive data can be exposed to multiple third parties, posing security and privacy issues.

Fortunately, Privacy Enhancing Technologies (PETs) have emerged to answer those issues. By using for instance Confidential Computing with secure enclaves, it becomes possible for data owners to have third parties manipulate their data in the Cloud, i.e. outside of the data owner infrastructure, without exposing the data to anyone else in clear.

While promising, finding easy to use and secure solutions is complicated, that is why we have built **BastionAI**, a confidential AI training framework to make these technologies.

## 🚀 Quick tour

BastionAI is a confidential training framework, built on top of PyTorch. 
It allows data owners, for instance hospitals, to upload their datasets inside a remote secure enclave, potentially in the Cloud, before datascientists send their models for training.

![BastionAI](assets/workflow_bastionai.png)

Once BastionAI has been deployed on a secure enclave, the workflow is simple:
- The data owner can upload their dataset to the remote enclave
- The data scientist defines locally her model before sending it to be trained on the remote enclave
- Once training is done, the model can be shared to parties authorized by the data owner, for instance the data scientist.

Note that the *data scientist never has access to the dataset in clear*, at best she can only get access to the weights of the models, but thanks to the use of DP during training, the information leakage on the initial training set is limited.

The *infra provider won't access neither the model nor the data* as secure enclaves provide end-to-end encryption. All data handled inside of the enclave are protected by hardware memory isolation and/or encryption.

### Finetune BERT on confidential data

We provide an example of how to fine-tune a BERT model on a small private dataset for classification of spams. One could imagine other interesting scenarios, such as medical or legal document analysis.

The notebook with detailled model and data preparation can be found here.

#### Data owner POV

First, the data owner will have to prepare his dataset before uploading to the BastionAI instance. 

```python
import bastionai

# Data preparation is detailled here
train_dataloader = ... 

with bastionai.Connection() as client:
    client.RemoteDataLoader(
      train_dataloader, 
      privacy_limit=8.0, 
      name="SMSSpamCollection"
    )
```

Once the PyTorch `DataLoader` is prepared, we simply need to connect to the remote enclave and upload the data. We first create an attested TLS channel with `bastionai.Connection()` that will connect securely.

Then we simply send the dataloader, and we define here a `privacy_limit` which is the maximum budget of Differential Privacy we allow data scientists to consume. DP consumption is tracked and if it is reached no further processing can be done on the data.

#### Data scientist POV

Now the data scientist just has to define her model before uploading it to the secure enclave and trigger the remote training of the model.

```python
import bastionai

# Model preparation is detailled here
model = ...

with bastionai.Connection() as client:
    remote_dataloader = client.get_available_datasets()[0]
    
    remote_learner = client.RemoteLearner(
      model,
      remote_dataloader,
      metric="cross_entropy",
      model_name="DistilBERT",
    )
    remote_learner.fit(nb_epochs=2, eps=2.0)
    trained_model = remote_learner.get_model()
```

Similarly we first connect to the secure enclave with `bastionai.Connection()`. Then, the data scientist needs to have a reference to the previously uploaded dataset. `client.get_available_datasets()` returns a list of uploaded dataloaders inside and selecting the first one returns a `RemoteDataLoader`. This object is a reference to a remote `DataLoader` hosted inside the enclave. Therefore the data scientist can only play with the reference but never has access to the dataset directly.

Once we have a local model, here a PyTorch `nn.Module`, and a `RemoteDataLoader`, we create a `RemoteLearner`, which will be used to train the model. This class is inspired from [fastai](https://docs.fast.ai/) `Learner` and has a similar high level interface. 

To trigger the remote training, we just need to call the `fit` method of our `RemoteLearner` and precise the number of epochs and DP budget we allow to consume. By using [Opacus](https://github.com/pytorch/opacus), the right parameters are computed automatically for the DP-Adam used for training.

Finally, once training is over, the data scientist can pull the trained model locally using `remote_learner.get_model()`.

## ✅ Key Features

- **Confidentiality made easy**: Easily train state-of-the-art models with confidentiality thanks to the use of secure enclaves and Differential Privacy.
- **Transparency**: Provide guarantees to third parties, for instance clients or regulators, that you are indeed providing **data protection**, through **code attestation**.
- **Extensible usage**: Explore different scenarios from confidential multi party training of cancer detection AI from a ResNet, to the fine-tuning of GPT models on confidential text, such as emails or documents.

## :question:FAQ

**Q: How do I make sure data that I send is protected**

**A:** We leverage secure enclaves to provide end-to-end protection. This means that even while your data is sent to someone else for them to apply an AI on it, your data remains protected thanks to hardware memory isolation and encryption.

We provide some information in our workshop [Reconcile AI and privacy with Confidential Computing](https://www.youtube.com/watch?v=tAT23GKMi_0).

You can also have a look on our series [Confidential Computing explained](https://blog.mithrilsecurity.io/confidential-computing-explained-part-1-introduction/).

**Q: Can I use Python script with BastionAI?**

**A:** We only support a `fastai`-style API, where you provide a `DataLoader` and a `nn.Module` through PyTorch. This is for both convenience and security, as allowing arbitrary computation inside an enclave can lead to [side-channel](https://en.wikipedia.org/wiki/Side-channel_attack) and leak information to the outside.

## Telemetry

BlindAI collects anonymous data regarding general usage, this allows us to understand how you are using the project. We only collect data regarding the execution mode (Hardware/Software) and the usage metrics.

This feature can be easily disabled, by settin up the environment variable `BLINDAI_DISABLE_TELEMETRY` to 1.

You can find more information about the telemetry in our [**documentation**](https://blindai.mithrilsecurity.io/en/latest/getting-started/telemetry/).

# :bangbang: Disclaimer

BastionAI is still in alpha and is being actively developed. It is provided as is, use it at your own risk.
