import torch
from bastionai.client import Connection, SGD
from bastionai.psg.nn import Linear
from torch import Tensor
from torch.nn import Module
from bastionai.utils import TensorDataset


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

X = torch.tensor([[0.1], [-1.0]])
Y = torch.tensor([[0.2], [-2.0]])
test_dataset = TensorDataset([X], Y)

with Connection("localhost", 50051) as client:
    remote_dataloader = client.RemoteDataset(
        train_dataset,
        test_dataset,
        name="1D Linear Regression",
        description="Dummy 1D Linear Regression Dataset (param is 2)",
        privacy_limit=8320.1,
    )

    remote_learner = client.RemoteLearner(
        lreg_model,
        remote_dataloader,
        max_batch_size=2,
        loss="l2",
        optimizer=SGD(lr=0.1, momentum=0.9),
        model_name="Linear 1x1",
        model_description="1D Linear Regression Model",
        expand=False,
    )

    print(f"Weight: {lreg_model.fc1.expanded_weight}")

    print(f"Devices: {client.get_available_devices()}")

    print(f"Optimizers: {(client.get_available_optimizers())}")

    remote_learner.fit(
        nb_epochs=200,
        eps=300.0,
        metric_eps=8000.0,
        per_epoch_checkpoint=True,
        per_n_step_checkpoint=2,
    )

    lreg_model = remote_learner.get_model()

    print(f"Weight: {lreg_model.fc1.expanded_weight}")

    remote_learner.test(metric_eps=20.0)
