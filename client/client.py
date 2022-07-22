from dataclasses import dataclass
from typing import Any, Iterator, List
from torch.nn import Module
from torch.utils.data import Dataset
from utils import serialize_dataset, serialize_model
from pb.remote_torch_pb2_grpc import RemoteTorchStub
from pb.remote_torch_pb2 import Chunk, Empty, Reference, TrainConfig
import grpc


@dataclass
class Client:
    stub: RemoteTorchStub

    def send_model(self, model: Module, description: str, secret: bytes) -> Reference:
        if not hasattr(model, "trainable_parameters"):
            raise Exception(
                "This model is not fully compatible with remote excecution on a BastionAI server. Consider using the @remote_module decorator.")
        if not hasattr(model, "grad_sample_parameters"):
            print("W: This model is not compatible with private optimizers, if you need DP guarantees, consider instantiating a PrivacyEngine and using the @engine.private_module(...) decorator.")
        return self.stub.SendModel(serialize_model(model, description=description, secret=secret))

    def send_dataset(self, dataset: Dataset, description: str, secret: bytes) -> Reference:

        return self.stub.SendDataset(serialize_dataset(dataset, description=description, secret=secret))

    def fetch_model(self, ref: Reference) -> Iterator[Chunk]:
        return self.stub.FetchModule(ref)

    def fetch_dataset(self, ref: Reference) -> Iterator[Chunk]:
        return self.stub.FetchDataset(ref)

    def get_available_models(self) -> List[Reference]:
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        return self.stub.AvailableDatasets(Empty())

    def train(self, config: TrainConfig) -> Reference:
        return self.stub.Train(config)

    def delete_dataset(self, ref: Reference) -> Empty:
        return self.stub.DeleteDataset(ref)

    def delete_module(self, ref: Reference) -> Empty:
        return self.stub.DeleteModule(ref)


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


if __name__ == '__main__':
    from dummy import DummyModel, DummyDataset
    model = DummyModel()
    dataset = DummyDataset()

    with Connection("localhost", 50051) as client:
        model_ref = client.send_model(model, "DummyModel", b"secret")
        print(f"Model ref: {model_ref}")
        # res = client.fetch_model(model_ref)

        res = client.send_dataset(dataset, "DummyDataset", b'secret')
        print(f"Dataset ref: {res}")

        res = client.delete_module(
            Reference(identifier="881379b7-0a24-4e00-9d3c-716b0b31f6b5"))
        # print(next(res))

        res = client.get_available_models()
        for model in res.list:  # type: ignore
            print(f"{model.identifier}, {model.description}")

        # dataset_ref = client.send_dataset(dataset, "DummyDataset")
        # print(f"Dataset ref: {dataset_ref}")

        # client.train(TrainConfig(model=model_ref, dataset=dataset_ref,
        #              batch_size=64, epochs=1, learning_rate=1e-3))
