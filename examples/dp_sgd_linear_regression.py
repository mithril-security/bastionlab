import torch
from bastionai.client import Connection, SGD
from bastionai.pb.remote_torch_pb2 import TestConfig, TrainConfig
from bastionai.psg.nn import Linear
from torch import Tensor
from torch.nn import Module
from bastionai.utils import TensorDataset
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

with Connection("localhost", 50051, default_secret=b"secret") as client:
    remote_dataloader = client.RemoteDataLoader(
        train_dataloader,
        test_dataloader,
        "Dummy 1D Linear Regression Dataset (param is 2)",
        'Linear Regression Dataset'
    )

    remote_learner = client.RemoteLearner(
        lreg_model,
        remote_dataloader,
        metric="l2",
        optimizer=SGD(lr=0.1),
        model_description="1D Linear Regression Model",
        model_name='lreg-model',
        expand=False,
    )

    print(f"Weight: {lreg_model.fc1.expanded_weight}")

    print(f"Devices: {client.get_available_devices()}")

    print(f"Optimizers: {(client.get_available_optimizers())}")

    remote_learner.fit(nb_epochs=100, eps=100.0)

    lreg_model = remote_learner.get_model()

    print(f"Weight: {lreg_model.fc1.expanded_weight}")

    remote_learner.test()
