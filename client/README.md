# BastionAI Client

BastionAI Client is a python library to create client applications for BastionAI Server (Mithril Security's confidential training server). 

**If you wish to know more about BastionAI, please have a look to the project [Github repository](https://github.com/mithril-security/bastionai/).**

## Installation

### Using pip
```bash
$ pip install bastionai
```

### Local installation
**Note**: It is preferrable to install BastionAI package in a virtual environment.

Execute the following command to install BastionAI locally.
```shell
pip install -e .
```

## Usage

### Uploading a model and datasets to BastionAI
The snippet below sets up a **very simple** linear regression model and dataset to train the model with.
```python
import torch
from bastionai.utils import TensorDataset  
from torch.nn import Module
from bastionai.psg.nn import Linear  
from torch.utils.data import DataLoader

class LReg(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(1, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(x)

lreg_model = LReg()

X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])
train_dataset = TensorDataset([X], Y)
train_dataloader = DataLoader(train_dataset, batch_size=2)

X = torch.tensor([[0.1], [-1.0]])
Y = torch.tensor([[0.2], [-2.0]])
test_dataset = TensorDataset([X], Y)
test_dataloader = DataLoader(test_dataset, batch_size=2)
```

### Training a model on BastionAI

With this snippet below, BastionAI is used to securely and remotely train the model. 

The model, along with the training and testing datasets are uploaded to BastionAI through the client API.

```python

from bastionai.client import Connection, SGD  


with Connection("localhost", 50051, default_secret=b"") as client:
    remote_dataloader = client.RemoteDataLoader(
        train_dataloader,
        test_dataloader,
        "Dummy 1D Linear Regression Dataset (param is 2)",
    )
    remote_learner = client.RemoteLearner(
        lreg_model,
        remote_dataloader,
        metric="l2",
        optimizer=SGD(lr=0.1),
        model_description="1D Linear Regression Model",
        expand=False,
    )

    remote_learner.fit(nb_epochs=100, eps=100.0)

    lreg_model = remote_learner.get_model() # Gets trained model from BastionAI server.
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under [Apache 2.0 License.](https://github.com/mithril-security/bastionai/blob/master/LICENSE)

