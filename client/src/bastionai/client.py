from dataclasses import dataclass
from typing import Any, List

import grpc
from remote_torch_pb2 import (Empty, Reference, TestConfig,
                                 TrainConfig, Devices)
from remote_torch_pb2_grpc import RemoteTorchStub
from torch.nn import Module
from torch.utils.data import Dataset

from bastionai.utils import (TensorDataset, dataset_from_chunks, deserialize_weights_to_model, metric_tqdm, metric_tqdm_with_epochs,
                   serialize_dataset, serialize_model)


@dataclass
class Client:
    stub: RemoteTorchStub

    def send_model(self, model: Module, description: str, secret: bytes, chunk_size=100_000_000) -> Reference:
        return self.stub.SendModel(serialize_model(model, description=description, secret=secret, chunk_size=chunk_size))

    def send_dataset(self, dataset: Dataset, description: str, secret: bytes, chunk_size=100_000_000, batch_size=1024) -> Reference:
        return self.stub.SendDataset(serialize_dataset(dataset, description=description, secret=secret, chunk_size=chunk_size, batch_size=batch_size))

    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        chunks = self.stub.FetchModule(ref)
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference) -> TensorDataset:
        return dataset_from_chunks(self.stub.FetchDataset(ref))

    def get_available_models(self) -> List[Reference]:
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        return self.stub.AvailableDatasets(Empty())

    def get_available_devices(self) -> List[Devices]:
        return self.stub.AvailableDevices(Empty())

    def train(self, config: TrainConfig) -> None:
        metric_tqdm_with_epochs(self.stub.Train(config), name=f"loss ({config.metric})")

    def test(self, config: TestConfig) -> float:
        metric_tqdm(self.stub.Test(config), name=f"metric ({config.metric})")

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
