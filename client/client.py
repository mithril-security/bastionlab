from dataclasses import dataclass
from typing import Any, Iterator, List
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset
from utils import deserialize_weights_to_model, serialize_dataset, serialize_model, ArtifactDataset
from pb.remote_torch_pb2_grpc import RemoteTorchStub
from pb.remote_torch_pb2 import Chunk, Empty, Reference, TrainConfig
import grpc


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


if __name__ == '__main__':
    from psg.nn import Linear
    from torch.nn import Module
    from torch import Tensor
    from torch.utils.data import Dataset
    import torch
    from typing import Tuple


    class LReg(Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = Linear(1, 1, 2)

        def forward(self, x: Tensor) -> Tensor:
            return self.fc1(x)


    class LRegDataset(Dataset):
        def __init__(self) -> None:
            super().__init__()
            self.X = torch.tensor([[0.0], [1.0], [0.5], [0.2]])
            self.Y = torch.tensor([[0.0], [2.0], [1.0], [0.4]])

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return (self.X[idx], self.Y[idx])


    model = LReg()
    dataset = LRegDataset()

    with Connection("localhost", 50051) as client:
        model_ref = client.send_model(model, "1D Linear Regression Model", b"secret")
        print(f"Model ref: {model_ref}")

        dataset_ref = client.send_dataset(dataset, "Dummy 1D Linear Regression Dataset (param is 2)", b'secret')
        print(f"Dataset ref: {dataset_ref}")

        model_list = client.get_available_models()
        for model in model_list.list:  # type: ignore
            print(f"{model.identifier}, {model.description}")

        # client.train(TrainConfig(
        #     model=model_ref,
        #     dataset=dataset_ref,
        #     batch_size=2,
        #     epochs=1,
        #     learning_rate=1e-1
        # ))
