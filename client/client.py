from dataclasses import dataclass
from typing import Any, Iterator, List
from torch.nn import Module
from torch.utils.data import Dataset
from utils import serialize_model, serialize_dataset
from pb.remote_torch_pb2_grpc import ReferenceProtocolStub
from pb.remote_torch_pb2 import AvailableObject, Chunk, Empty, Reference, TrainConfig
import grpc


@dataclass
class Client:
    stub: ReferenceProtocolStub

    def send_model(self, model: Module, description: str) -> Reference:
        if not hasattr(model, "trainable_parameters"):
            raise Exception(
                "This model is not fully compatible with remote excecution on a BastionAI server. Consider using the @remote_module decorator.")
        if not hasattr(model, "grad_sample_parameters"):
            print("W: This model is not compatible with private optimizers, if you need DP guarantees, consider instantiating a PrivacyEngine and using the @engine.private_module(...) decorator.")
        return self.stub.SendModel(serialize_model(model, description=description))

    def send_dataset(self, dataset: Dataset, description: str) -> Reference:
        return self.stub.SendData(serialize_dataset(dataset, description=description))

    def fetch_model(self, ref: Reference) -> Iterator[Chunk]:
        return self.stub.Fetch(ref)

    def get_available_models(self) -> List[AvailableObject]:
        return self.stub.GetAvailableModels(Empty())

    def get_available_datasets(self) -> List[AvailableObject]:
        return self.stub.GetAvailableDataSets(Empty())

    def train(self, config: TrainConfig) -> Reference:
        return self.stub.Train(config)


@dataclass
class Connection:
    host: str
    port: int
    channel: Any = None

    def __enter__(self) -> Client:
        self.channel = grpc.insecure_channel("localhost:50051")
        return Client(ReferenceProtocolStub(self.channel))

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.channel.close()


if __name__ == '__main__':
    from dummy import DummyModel, DummyDataset
    model = DummyModel()
    dataset = DummyDataset()

    with Connection("localhost", 50051) as client:
        model_ref = client.send_model(model, "DummyModel")
        print(f"Model ref: {model_ref}")
        # res = client.fetch_model(model_ref)

        res = client.get_available_models()
        for model in res.available_objects: # type: ignore
            print(f"{model.reference}, {model.description}")

        dataset_ref = client.send_dataset(dataset, "DummyDataset")
        print(f"Dataset ref: {dataset_ref}")

        client.train(TrainConfig(model=model_ref, dataset=dataset_ref, batch_size=64, epochs=1, learning_rate=1e-3))

