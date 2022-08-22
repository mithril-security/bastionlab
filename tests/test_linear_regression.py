import subprocess
import os
from typing import Tuple
import unittest
from torch import Tensor
import torch
from bastionai.client import Connection
from torch.nn import Module
from bastionai.psg.nn import Linear
from torch.utils.data import Dataset
import logging


from server import launch_server

logging.basicConfig(level=logging.INFO)


class LinRegTest(unittest.TestCase):

    def test_model_upload(self):
        with Connection('localhost', 50051) as client:
            ref = client.send_model(
                model, "A simple linear regression model", b"secret")

        self.assertEqual(ref.description, "A simple linear regression model")

    def test_dataset_upload(self):
        with Connection('localhost', 50051) as client:
            ref = client.send_dataset(
                dataset, "A simple linear regression model", b"secret")

        self.assertEqual(ref.description, "A simple linear regression model")


model, dataset = None, None


def setUpModule():
    global model, dataset

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

    model = LReg()
    dataset = LRegDataset()


if __name__ == '__main__':
    unittest.main()
