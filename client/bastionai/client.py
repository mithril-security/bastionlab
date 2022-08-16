from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import grpc
from remote_torch_pb2 import Empty, Reference, TestConfig, TrainConfig, Devices
from remote_torch_pb2_grpc import RemoteTorchStub
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import numpy as np

from bastionai.utils import (
    TensorDataset,
    dataset_from_chunks,
    deserialize_weights_to_model,
    metric_tqdm,
    metric_tqdm_with_epochs,
    serialize_dataset,
    serialize_model,
)
from bastionai.psg import expand_weights
from bastionai.optimizer_config import *

if TYPE_CHECKING:
    from bastionai.learner import RemoteLearner, RemoteDataLoader


@dataclass
class Client:
    """BastionAI client class."""

    stub: RemoteTorchStub
    default_secret: bytes

    def send_model(
        self, model: Module, description: str, secret: bytes, chunk_size=100_000_000
    ) -> Reference:
        """Uploads Pytorch Modules to BastionAI

        This endpoint transforms Pytorch modules into TorchScript modules and sends
        them to the BastionAI server.

        Args:
            model (Module):
                This is Pytorch's nn.Module.
            description (str):
                A string description of the module being uploaded.
            secret (bytes):
                User secret to secure module with.

        Returns:
            Reference:
                BastionAI reference object
        """
        return self.stub.SendModel(
            serialize_model(
                model, description=description, secret=secret, chunk_size=chunk_size
            )
        )

    def send_dataset(
        self,
        dataset: Dataset,
        description: str,
        secret: bytes,
        chunk_size=100_000_000,
        batch_size=1024,
    ) -> Reference:
        """Uploads Pytorch Dataset to BastionAI.

        Args:
            model (Module):
                This is Pytorch's nn.Module.
            description (str):
                A string description of the dataset being uploaded.
            secret (bytes):
                User secret to secure module with.

        Returns:
            Reference:
                BastionAI reference object
        """
        return self.stub.SendDataset(
            serialize_dataset(
                dataset,
                description=description,
                secret=secret,
                chunk_size=chunk_size,
                batch_size=batch_size,
            )
        )

    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        """Fetchs the weights of a trained model with a BastionAI reference.

        Args:
            model (Module):
                This is Pytorch's nn.Module corresponding to an uploaded module.
            ref (Reference):
                BastionAI reference object corresponding to a module.
        """
        chunks = self.stub.FetchModule(ref)
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference) -> TensorDataset:
        """Fetchs the dataset with a BastionAI reference.

        Args:
            ref (Reference):
                BastionAI reference object corresponding to a dataset.

        Returns:
            ArtifactDataset: A wrapper to convert Tensors from BastionAI to Pytorch DataLoader
        """
        return dataset_from_chunks(self.stub.FetchDataset(ref))

    def get_available_models(self) -> List[Reference]:
        """Gets a list of references of available models on BastionAI.

        Returns:
            List[Reference]: A list of BastionAI available model references.
        """
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        """Gets a list of references of datasets on BastionAI.

        Returns:
            List[Reference]:
                A list of BastionAI available dataset references.
        """
        return self.stub.AvailableDatasets(Empty())

    def get_available_devices(self) -> List[str]:
        """Gets a list of devices available on BastionAI.
        Returns:

            List[Devices]:
                A list of BastionAI available device references.
        """
        return self.stub.AvailableDevices(Empty()).list

    def get_available_optimizers(self) -> List[str]:
        """Gets a list of optimizers supported by BastionAI.

        Returns:
            List[Optimizers]:
                A list of optimizers available on BastionAI training server.
        """
        return self.stub.AvailableOptimizers(Empty()).list

    def train(self, config: TrainConfig) -> None:
        """Trains a model with `TrainConfig` configuration on BastionAI.

        Args:
            config (TrainConfig):
                Training configuration to pass to BastionAI.
        """
        metric_tqdm_with_epochs(self.stub.Train(config), name=f"loss ({config.metric})")

    def test(self, config: TestConfig) -> float:
        """Tests a dataset on a model on BastionAI.

        Args:
            config (TestConfig): Configuration for testing on BastionAI.

        Returns:
            float:
                The evaluation of the model on the datatset.
        """
        metric_tqdm(self.stub.Test(config), name=f"metric ({config.metric})")

    def delete_dataset(self, ref: Reference) -> None:
        """Deletes a dataset with reference on BastionAI.

        Args:
            ref (Reference): BastionAI reference to dataset.

        """
        self.stub.DeleteDataset(ref)

    def delete_module(self, ref: Reference) -> None:
        """Deletes a model with reference from BastionAI.

        Args:
            ref (Reference): BastionAI reference to dataset.
        """
        self.stub.DeleteModule(ref)

    def RemoteDataLoader(self, *args, **kwargs) -> "RemoteDataLoader":
        from bastionai.learner import RemoteDataLoader

        return RemoteDataLoader(self, *args, **kwargs)

    def RemoteLearner(self, *args, **kwargs) -> "RemoteLearner":
        from bastionai.learner import RemoteLearner

        return RemoteLearner(self, *args, **kwargs)


@dataclass
class Connection:
    """Connection class for creating connections to BastionAI."""

    host: str
    port: int
    default_secret: bytes
    channel: Any = None

    def __enter__(self) -> Client:
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        return Client(RemoteTorchStub(self.channel), self.default_secret)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.channel.close()
