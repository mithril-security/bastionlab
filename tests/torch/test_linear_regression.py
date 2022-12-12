import logging
import unittest
from ..server import launch_server  # type: ignore [import]
import torch
from bastionlab import Connection  # type: ignore [import]
from bastionlab.torch.optimizer_config import SGD  # type: ignore [import]
from bastionlab.torch.psg.nn import Linear  # type: ignore [import]
from bastionlab.torch.utils import TensorDataset  # type: ignore [import]

logging.basicConfig(level=logging.INFO)


class LinRegTest(unittest.TestCase):
    def test_model_and_data_upload(self):
        client = Connection("localhost").client.torch
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
        client = Connection("localhost").client.torch
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


def setUpModule():
    import os

    global train_dataset, test_dataset, lreg_model

    LIBTORCH = os.environ.get("LIBTORCH_")
    LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH_")
    launch_server(
        {
            "LIBTORCH": LIBTORCH,
            "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
            "DISABLE_AUTHENTICATION": "1",
            "BASTIONLAB_DISABLE_TELEMETRY": "1",
        }
    )

    class LReg(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = Linear(1, 1, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
