## Installing BastionAI Client

### Via pip

```
pip install bastionai
```

### From source

First clone our repo:
```
git clone https://github.com/mithril-security/bastionai.git
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
#overview of dataset upload process

import bastionai

# Data preparation to be detailled here
train_dataset = ... 

# Server/port details for connection
host = "localhost"
port = 50051

with bastionai.Connection(host, port) as client:
    client.RemoteDataset(
        train_dataset, 
        privacy_limit=6.0, 
        name="SMSSpamCollection"
    )
```

The dataset should simply be a Pytorch [`Dataset`](https://pytorch.org/vision/stable/datasets.html) instance. The Data Owner connects to the remote server running inside the TEE and uploads the dataset. Under the hood, `bastionai.Connection()` secures the transfer with an attested TLS channel.

The Data Owner has the opportunity to set a `privacy_limit` which is the maximum Differential Privacy budget they allow the Data Scientist to consume. DP consumption is directly tracked by the code running inside the TEE for increased security gurantees. When the limit is reached the server does not allow any further processing.

#### Data scientist POV

The Data Scientist defines their model locally, sends it to the TEE and triggers training of the model on the owner's data.

```python
#overiew of model upload and training process

import bastionai

# Model preparation to be detailled here
model = ...

# Server/port details for connection
host = "localhost"
port = 50051

with bastionai.Connection(host, port) as client:
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

To trigger training with the previously defined (hyper)parameters, the Data Scientist calls the `fit` method. They pass the desired number of epochs and DP budget and also have the opportunity to override some of the previously defined (hyper)parameters. Note that the DP budget is global for the whole training procedure: our internal private learning library automatically computes the right per-step DP-SGD parameters.

Finally, once training is over, the Data Scientist may pull the trained model using the `get_model` method (if they have the permissions to do so).

## Installing BastionAI Server

### Using our official Docker image

```
docker pull mithrilsecuritysas/bastionai
docker run -p 50051:50051 -d mithrilsecuritysas/bastionai:latest
```

### By locally building our Docker image

Clone our repository and build the image using our Dockerfile:
```
git clone https://github.com/mithril-security/bastionai.git
cd ./bastionai/server
docker build -t bastionai:0.1.2 -t bastionai:latest .
```
Then simply run a container based on the image:
```
docker run -p 50051:50051 -d bastionai
```

### From source

First make sure that the following build dependences (Debian-like systems) are installed on your machine:
```
sudo apt update && apt install -y build-essential patchelf libssl-dev pkg-config curl unzip
```

Then, clone our repository:
```
git clone https://github.com/mithril-security/bastionai.git
```
Download and unzip libtorch (Pytorch's C++ backend) from [Pytorch's website](https://pytorch.org/) (you can chose the right build according to your cuda version):
```
cd ./bastionai
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch.zip
```
Lib torch binaries are now available under the libtorch folder. You can now turn to building the server crates:
```
cd server
LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make compile
make copy-bin
make init
```

To run the server, simply use:
```
make run
```
