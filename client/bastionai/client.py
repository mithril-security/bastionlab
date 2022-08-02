from dataclasses import dataclass
from typing import Any, Iterator, List

import grpc
from pb.remote_torch_pb2 import (Chunk, Empty, Reference, TestConfig,
                                 TrainConfig)
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
        return self.stub.SendModel(serialize_model(model, description=description, secret=secret))

    def send_dataset(self, dataset: Dataset, description: str, secret: bytes) -> Reference:
        return self.stub.SendDataset(serialize_dataset(dataset, description=description, secret=secret))

    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        chunks = self.stub.FetchModule(ref)
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference) -> ArtifactDataset:
        return ArtifactDataset(self.stub.FetchDataset(ref))

    def get_available_models(self) -> List[Reference]:
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        return self.stub.AvailableDatasets(Empty())

    def train(self, config: TrainConfig) -> None:
        self.stub.Train(config)

    def test(self, config: TestConfig) -> float:
        return self.stub.Test(config)

    def delete_dataset(self, ref: Reference) -> None:
        self.stub.DeleteDataset(ref)

    def delete_module(self, ref: Reference) -> None:
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
