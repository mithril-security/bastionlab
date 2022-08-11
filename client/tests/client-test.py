from typing import Tuple

import torch
from bastionai.pb.remote_torch_pb2 import TestConfig, TrainConfig
from bastionai.psg.nn import Linear
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from bastionai.utils.utils import create_training_config, create_test_config, Optimizers

from bastionai.client import connect


class LReg(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(1, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(x)


class LRegDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
        self.Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.X[idx], self.Y[idx])


lreg_model = LReg()
lreg_dataset = LRegDataset()

client = connect(addr='localhost', port=50053)

model_ref = client.send_model(
    lreg_model, "1D Linear Regression Model", b"secret")
print(f"Model ref: {model_ref}")

dataset_ref = client.send_dataset(
    lreg_dataset, "Dummy 1D Linear Regression Dataset (param is 2)", b'secret')
print(f"Dataset ref: {dataset_ref}")

print(f"Weight: {lreg_model.fc1.inner.expanded_weight}")
print(f"Devices: {client.get_available_devices()}")

print(f"Optimizers: {(client.get_available_optimizers())}")

client.train(
    create_training_config(model_ref,
                           dataset_ref,
                           batch_size=2,
                           epochs=100,
                           learning_rate=0.1,
                           weight_decay=0.,
                           noise_multiplier=0.1,
                           max_grad_norm=1.,
                           extra_args={
                               "momentum": 0.,
                               "dampening": 0.,
                               "nesterov": False
                           },
                           optimizer=Optimizers.SGD))

client.fetch_model_weights(lreg_model, model_ref)
print(f"Weight: {lreg_model.fc1.inner.expanded_weight}")

client.test(create_test_config(model_ref, dataset_ref, 2))
