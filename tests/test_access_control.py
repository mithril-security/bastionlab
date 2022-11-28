import subprocess
import os
from typing import Tuple
import unittest
from torch import Tensor
import torch
from bastionai.client import Connection, SGD  # type: ignore [import]
from torch.nn import Module
from bastionai.psg.nn import Linear  # type: ignore [import]
from torch.utils.data import DataLoader

import logging
from bastionai.utils import TensorDataset  # type: ignore [import]
import bastionai.errors
from bastionai import SigningKey, PublicKey, LicenseBuilder, Reference
from server import launch_server  # type: ignore [import]

logging.basicConfig(level=logging.DEBUG)

data_owner_key = SigningKey.generate()
data_scientist_key = SigningKey.generate()


class LinRegTest(unittest.TestCase):
    def test_failing_pbk(self):
        with Connection("localhost", 50051, license_key=data_scientist_key) as data_scientist:
            with Connection("localhost", 50051, license_key=data_owner_key) as data_owner:
                # not listable

                dataset = data_owner.RemoteDataset(
                    train_dataset,
                    name="1D Linear Regression 1",
                    description="Dummy 1D Linear Regression Dataset (param is 2)",
                    privacy_limit=8320.1,
                    license=(
                        # By default, the data owner should be able to do anything it wants with the model
                        LicenseBuilder.default_with_pubkey(data_owner_key.pubkey)
                        # Add a pubkey that can train using this dataset
                        # .trainable(signed_with=data_scientist_pubkey) # NOT trainable
                        # Add a pubkey that can list this dataset (client.available_datasets())
                        # .listable(signed_with=data_scientist_pubkey) # NOT listable
                        .build()
                    ),
                )

                datasets = data_scientist.get_available_datasets()
                ds = [el for el in datasets if el.name == "1D Linear Regression 1"]
                self.assertEqual(len(ds), 0)

                data_owner.delete_dataset(dataset.train_dataset_ref)

                # listable not trainable

                dataset = data_owner.RemoteDataset(
                    train_dataset,
                    name="1D Linear Regression 2",
                    description="Dummy 1D Linear Regression Dataset (param is 2)",
                    privacy_limit=8320.1,
                    license=(
                        # By default, the data owner should be able to do anything it wants with the model
                        LicenseBuilder.default_with_pubkey(data_owner_key.pubkey)
                        # Add a pubkey that can train using this dataset
                        # .trainable(signed_with=data_scientist_pubkey) # NOT trainable
                        # Add a pubkey that can list this dataset (client.available_datasets())
                        .listable(signed_with=data_scientist_key.pubkey)
                        .build()
                    ),
                )

                datasets = data_scientist.get_available_datasets()
                ds = [el for el in datasets if el.name == "1D Linear Regression 2"]
                self.assertEqual(len(ds), 1)
                remote_learner = data_scientist.RemoteLearner(
                    lreg_model,
                    ds[0],
                    loss="l2",
                    optimizer=SGD(lr=0.1),
                    model_name="Linear 1x1",
                    model_description="1D Linear Regression Model",
                    expand=False,
                    max_batch_size=2,
                )
                with self.assertRaises(bastionai.errors.GRPCException) as cm:
                    remote_learner.fit(nb_epochs=2, eps=2.0)


def setUpModule():
    global train_dataset, test_dataset, lreg_model

    launch_server()

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


if __name__ == "__main__":
    unittest.main()
