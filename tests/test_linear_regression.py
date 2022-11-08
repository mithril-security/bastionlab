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
from bastionai import SigningKey, PublicKey, LicenseBuilder, Reference
from server import launch_server  # type: ignore [import]

logging.basicConfig(level=logging.DEBUG)

data_owner_key = SigningKey.from_pem_or_generate("./data_owner.key.pem")
data_scientist_key = SigningKey.from_pem_or_generate("./data_scientist.key.pem")


class LinRegTest(unittest.TestCase):
    def test_model_and_data_upload(self):
        with Connection("localhost", 50051, license_key=data_owner_key) as client:
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
                loss="l2",
                optimizer=SGD(lr=0.1),
                model_name="Linear 1x1",
                model_description="1D Linear Regression Model",
                expand=False,
                max_batch_size=2,
            )

        self.assertEqual(remote_learner.client, client)

    def test_weights_before_and_after_upload(self):
        with Connection("localhost", 50051, license_key=data_owner_key) as client:
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
                loss="l2",
                optimizer=SGD(lr=0.1),
                model_name="Linear 1x1",
                model_description="1D Linear Regression Model",
                expand=False,
                max_batch_size=2,
            )

            bastion_lreg_model = remote_learner.get_model()
        self.assertEqual(bastion_lreg_model, lreg_model)

    def test_data_owner_data_scientist(self):
        # Test the whole workflow

        data_scientist_key = SigningKey.from_pem_or_generate("./data_scientist.key.pem")
        data_owner_key = SigningKey.from_pem_or_generate("./data_owner.key.pem")
        data_scientist_key.pubkey.save_pem("./data_scientist.pem")

        # DATA OWNER POV
        data_owner_key = SigningKey.from_pem("./data_owner.key.pem")
        data_scientist_pubkey = PublicKey.from_pem("./data_scientist.pem")
        with Connection("localhost", 50051, license_key=data_owner_key) as client:
            dataset = client.RemoteDataset(
                train_dataset,
                name="1D Linear Regression",
                description="Dummy 1D Linear Regression Dataset (param is 2)",
                privacy_limit=8320.1,
                license=(
                    # By default, the data owner should be able to do anything it wants with the model
                    LicenseBuilder.default_with_pubkey(data_owner_key.pubkey)
                    # Add a pubkey that can train using this dataset
                    .trainable(signed_with=data_scientist_pubkey)
                    # Add a pubkey that can list this dataset (client.available_datasets())
                    .listable(signed_with=data_scientist_pubkey)
                    .build()
                ),
            )
        train_dataset_hash = dataset.train_dataset_ref.hash

        # DATA SCIENTIST POV
        data_scientist_key = SigningKey.from_pem("./data_owner.key.pem")
        data_scientist_pubkey = PublicKey.from_pem("./data_scientist.pem")
        # train_dataset_hash = ""
        with Connection("localhost", 50051, license_key=data_scientist_key) as client:
            remote_learner = client.RemoteLearner(
                lreg_model,
                Reference.from_hash(train_dataset_hash),
                loss="l2",
                optimizer=SGD(lr=0.1),
                model_name="Linear 1x1",
                model_description="1D Linear Regression Model",
                expand=False,
                max_batch_size=2,
            )

            remote_learner.fit(nb_epochs=2, eps=2.0)
            trained_model = remote_learner.get_model()



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
