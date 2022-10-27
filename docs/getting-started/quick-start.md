## Installing BastionAI Client

BastionAI can easily be installed either through Pypi or you can build the client yourself from source.

### Install via pip

```shell
pip install bastionai
```

This package is enough for the deployment and querying of models on our managed infrastructure. For on-premise deployment, you will have to deploy our Docker image, you can read how to deploy it [here](../deployment/on_premise.md).

### Build from source

First clone our repo:
```
git clone git@github.com:mithril-security/bastionai.git
```
Then install the client library:
```
cd ./bastionai/client
make install
```

## Python API Walkthough: fine-tuning BERT on confidential data

We provide an example of how to privately fine-tune a BERT model on a small dataset for spam/ham classification.
In the following, we assume the open source dataset we use should remain private.
This simple example may be extended to more complex scenarios, such as medical or legal document analysis.

The notebook with detailled model and data preparation can be found [here](https://github.com/mithril-security/bastionai/blob/master/examples/distilbert_example_notebook.ipynb).

#### Data owner POV

The Data Owner prepares their dataset before uploading it to the BastionAI server. 

```python
import bastionai

# Data preparation is detailled here
train_dataset = ... 

with bastionai.Connection() as client:
    client.RemoteDataset(
        train_dataset, 
        privacy_limit=6.0, 
        name="SMSSpamCollection"
    )
```

The dataset is simply a Pytorch [`Dataset`](https://pytorch.org/vision/stable/datasets.html) instance. The Data Owner connects to the remote server running inside the TEE and uploads the dataset. Under the hood, `bastionai.Connection()` secures the transfert with an attested TLS channel.

The Data Owner has the opportunity to set a `privacy_limit` which is the maximum Differential Privacy budget they allow the Data Scientist to consume. DP consumption is directly tracked by the code running inside the TEE for increased security gurantees. When the limit is reached the server does not allow any further processing.

#### Data scientist POV

The Data Scientist defines their model locally, sends it to the TEE and triggers training of the model on the owner's data.

```python
import bastionai

# Model preparation is detailled here
model = ...

with bastionai.Connection() as client:
    remote_dataset = client.list_remote_datasets()[0]
    
    remote_learner = client.RemoteLearner(
        model,
        remote_dataset,
        max_batch_size=4,
        loss="cross_entropy",
        optimizer=Adam(lr=5e-5),
        model_name="DistilBERT",
    )
    remote_learner.fit(nb_epochs=2, eps=2.0)
    trained_model = remote_learner.get_model()
```

Similarly, the Data Scientist connects to the TEE with `bastionai.Connection()`. They first query the server to get the list of remotely available datasets and choose the one they like. `client.get_available_datasets()` returns a list of `RemoteDataset` objects that are references to a remote dataset located in the TEE. This means the Data Scientist can play with the reference but never has direct access to the data.

Then, the Data Scientist bundles their local PyTorch's `nn.Module` model with all the (hyper)parameters necessary to train it on the specified `RemoteDataset`. Under the hood, the `RemoteLearner` class serializes the model in [TorchScript](https://pytorch.org/docs/stable/jit.html) and sends it to the TEE. This class was inspired by [fastai](https://docs.fast.ai/) `Learner` and has a similar high level interface. The use of TorchScript for serialization improves security by allowing the Data Scientist to execute a limited set of operations on the data.

To trigger training with the previously defined (hyper)parameters, the Data Scientist calls the `fit` method. They pass the desired number of epochs and DP budget and also have the opprotunity to override some of the previously defined (hyper)parameters. Note that the DP budget is global for the whole training procedure: our internal private learning library automatically computes the right per-step DP-SGD parameters.

Finally, once training is over, the Data Scientist may pull the trained model using the `get_model` method (if they are allowed to).

## Installing BastionAI Server

You can find out how to install and deploy BastionAI server right [here](../deployment/on_premise.md).