from dataclasses import dataclass
from typing import Any
from torch.nn import Module
from torch.utils.data import Dataset
from utils import serialize_model, serialize_dataset
from remote_torch_pb2_grpc import ReferenceProtocolStub
from remote_torch_pb2 import Reference
import grpc

@dataclass
class Client:
    stub: ReferenceProtocolStub

    def send_model(self, model: Module) -> Reference:
        if not hasattr(model, "trainable_parameters"):
            raise Exception("This model is not fully compatible with remote excecution on a BastionAI server. Consider using the @remote_module decorator.")
        if not hasattr(model, "grad_sample_parameters"):
            print("W: This model is not compatible with private optimizers, if you need DP guarantees, consider instantiating a PrivacyEngine and using the @engine.private_module(...) decorator.")
        return self.stub.SendModel(serialize_model(model))
    
    def send_dataset(self, dataset: Dataset) -> Reference:
        return self.stub.SendData(serialize_dataset(dataset))

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
    from dummy import DummyModel
    model = DummyModel()

    with Connection("localhost", 50051) as client:
        client.send_model(model)

