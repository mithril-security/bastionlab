from dataclasses import dataclass
from typing import Any, Iterator, List

import grpc
from pb.remote_torch_pb2 import (Chunk, Empty, Reference, TestConfig,
                                 TrainConfig, Devices)
from pb.remote_torch_pb2_grpc import RemoteTorchStub
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from utils import (ArtifactDataset, deserialize_weights_to_model, metric_tqdm, metric_tqdm_with_epochs,
                   serialize_dataset, serialize_model)
from tqdm import tqdm
from time import sleep


@dataclass
class Client:
    """
        BastionAI client class.
    """
    stub: RemoteTorchStub

    def send_model(self, model: Module, description: str, secret: bytes) -> Reference:
        """
        This method provides the endpoint to uploading Pytorch Modules to BastionAI.

        Args:
            model (Module): This is Pytorch's nn.Module.
            description (str): A string description of the module being uploaded.
            secret (bytes): User secret to secure module with.

        Returns:
            Reference: BastionAI reference object
        """
        return self.stub.SendModel(serialize_model(model, description=description, secret=secret))

    def send_dataset(self, dataset: Dataset, description: str, secret: bytes) -> Reference:
        """
        This method provides the endpoint for uploading Pytorch DataLoader to BastionAI.

        Args:
            model (Module): This is Pytorch's nn.Module.
            description (str): A string description of the module being uploaded.
            secret (bytes): User secret to secure module with.

        Returns:
            Reference: BastionAI reference object
        """
        return self.stub.SendDataset(serialize_dataset(dataset, description=description, secret=secret))

    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        """
        This method provides the endpoint for fetching the weights of a trained model with a BastionAI reference.

        Args:
            model (Module): This is Pytorch's nn.Module corresponding to an uploaded module.
            ref (Reference): BastionAI reference object corresponding to a module.

        Returns:
            None
        """
        chunks = self.stub.FetchModule(ref)
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference) -> ArtifactDataset:
        """
        This method provides the endpoint for fetching the dataset with a BastionAI reference.

        Args:
            ref (Reference): BastionAI reference object corresponding to a dataset.

        Returns:
            ArtifactDataset: A wrapper to convert Tensors from BastionAI to Pytorch DataLoader
        """
        return ArtifactDataset(self.stub.FetchDataset(ref))

    def get_available_models(self) -> List[Reference]:
        """
            This method provides the endpoint for getting a list of references of available models on BastionAI.

            Returns: 
                List[Reference]: A list of BastionAI available model references.
        """
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        """
            This method provides the endpoint for getting a list of references of datasets on BastionAI.

            Returns: 
                List[Reference]: A list of BastionAI available dataset references.
        """
        return self.stub.AvailableDatasets(Empty())

    def get_available_devices(self) -> List[Devices]:
        """
            This method provides the endpoint for getting a list of devices available on BastionAI.

            Returns: 
                List[Reference]: A list of BastionAI available device references.
        """
        return self.stub.AvailableDevices(Empty())

    def train(self, config: TrainConfig) -> None:
        """
        This method provides the endpoint for training a model with `TrainConfig` configuration on BastionAI.

        Args:
            config (TrainConfig): Training configuration to pass to BastionAI.
        Returns:
            None
        """
        metric_tqdm_with_epochs(self.stub.Train(config), name=f"loss ({config.metric})")

    def test(self, config: TestConfig) -> float:
        """
        This method provides the endpoint for testing a dataset on a model on BastionAI.

        Args:
            config (TestConfig): Configuration for testing on BastionAI.
        
        Returns:
            float
        """
        metric_tqdm(self.stub.Test(config), name=f"metric ({config.metric})")

    def delete_dataset(self, ref: Reference) -> None:
        """
            This method provides the endpoint for deleting a dataset with reference on BastionAI.

            Args:
                ref (Reference): BastionAI reference to dataset.

            Returns:
                None
        """
        self.stub.DeleteDataset(ref)

    def delete_module(self, ref: Reference) -> None:
        """
            This method provides the endpoint for deleting a model with reference from BastionAI.

            Args:
                ref (Reference): BastionAI reference to dataset.

            Returns:
                None
        """
        self.stub.DeleteModule(ref)


@dataclass
class Connection:
    """
    Connection class for creating connections to BastionAI.
    """
    host: str
    port: int
    channel: Any = None

    def __enter__(self) -> Client:
        self.channel = grpc.insecure_channel("localhost:50051")
        return Client(RemoteTorchStub(self.channel))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.channel.close()
