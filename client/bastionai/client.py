from dataclasses import dataclass
from typing import Any, Iterator, List

import grpc
from pb.remote_torch_pb2 import (Chunk, Empty, Reference, TestConfig,
                                 TrainConfig, Devices)
from pb.remote_torch_pb2_grpc import RemoteTorchStub
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from utils import (ArtifactDataset, deserialize_weights_to_model,
                   serialize_dataset, serialize_model)


@dataclass
class Client:
    stub: RemoteTorchStub

    def send_model(self, model: Module, description: str, secret: bytes) -> Reference:
        """
        send_model provides the endpoint to uploading Pytorch Modules to BastionAI.

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
        send_data provides the endpoint for uploading Pytorch DataLoader to BastionAI.

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
        Args:
            ref (Reference): BastionAI reference object corresponding to a dataset.

        Returns:
            ArtifactDataset: A wrapper to convert Tensors from BastionAI to Pytorch DataLoader
        """
        return ArtifactDataset(self.stub.FetchDataset(ref))

    def get_available_models(self) -> List[Reference]:
        """
            Returns: 
                List[Reference]: A list of BastionAI available model references.
        """
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        """
            Returns: 
                List[Reference]: A list of BastionAI available dataset references.
        """
        return self.stub.AvailableDatasets(Empty())

    def get_available_devices(self) -> List[Devices]:
        """
            Returns: 
                List[Reference]: A list of BastionAI available device references.
        """
        return self.stub.AvailableDevices(Empty())

    def train(self, config: TrainConfig) -> None:
        """
        Trains model with `TrainConfig` on BastionAI server.

        Args:
            config (TrainConfig): Training configuration to pass to BastionAI.
        Returns:
            None
        """
        self.stub.Train(config)

    def test(self, config: TestConfig) -> float:
        """
            
        """
        return self.stub.Test(config)

    def delete_dataset(self, ref: Reference) -> None:
        """
            Delets dataset with reference from BastionAI server.

            Args:
                ref (Reference): BastionAI reference to dataset.

            Returns:
                None
        """
        self.stub.DeleteDataset(ref)

    def delete_module(self, ref: Reference) -> None:
        """
            Delets model with reference from BastionAI server.

            Args:
                ref (Reference): BastionAI reference to dataset.

            Returns:
                None
        """
        self.stub.DeleteModule(ref)


@dataclass
class Connection:
    host: str
    port: int
    channel: Any = None

    def __enter__(self) -> Client:
        self.channel = grpc.insecure_channel("localhost:50051")
        return Client(RemoteTorchStub(self.channel))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.channel.close()
