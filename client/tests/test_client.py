from typing import Iterator

from bastionai.pb.remote_torch_pb2 import Chunk, Reference, TrainConfig, Metric
from bastionai.utils import *
from bastionai.client import Client
from test_utils import Params, DummyModule, module_eq, simple_dataset
from torch.utils.data import DataLoader


class MockStub:
    def __init__(self):
        self.data_chunks = []
        self.counter = 0
        self.client_info = ClientInfo()

    def _store(self, chunks: Iterator[Chunk]) -> Reference:
        self.data_chunks.append(chunks)
        ref = Reference(identifier=f"{self.counter}", description="")
        self.counter += 1
        return ref

    def SendModel(self, chunks: Iterator[Chunk]) -> Reference:
        model = list(
            unstream_artifacts(
                (chunk.data for chunk in chunks), deserialization_fn=torch.jit.load
            )
        )[0]
        chunks = serialize_model(Params(model), "", "", b"", self.client_info)
        return self._store(chunks)

    def SendDataset(self, chunks: Iterator[Chunk]) -> Reference:
        return self._store(chunks)

    def FetchModule(self, ref: Reference) -> Iterator[Chunk]:
        return self.data_chunks[int(ref.identifier)]

    def FetchDataset(self, ref: Reference) -> Iterator[Chunk]:
        return self.data_chunks[int(ref.identifier)]

    def Train(self, config: TrainConfig) -> Reference:
        return Reference(identifier="0", description="run 0")

    def Test(self, config: TrainConfig) -> Reference:
        return Reference(identifier="0", description="run 0")

    def GetMetric(self, run: Reference) -> Metric:
        return Metric(value=0.0, batch=0, epoch=0, nb_epochs=1, nb_batches=1)


def test_api(simple_dataset):
    model = DummyModule()

    client = Client(MockStub(), client_info=ClientInfo())

    dl = DataLoader(simple_dataset, batch_size=2)
    remote_dataloader = client.RemoteDataLoader(dl, dl, privacy_limit=344.0)

    remote_learner = client.RemoteLearner(
        model, remote_dataloader, loss="l2", expand=False, progress=False
    )

    remote_learner.fit(nb_epochs=100, eps=142.0)
    remote_learner.test()
    assert (
        remote_learner.log
        == [
            Metric(
                value=0.0, batch=0, epoch=0, nb_epochs=1, nb_batches=1
            )
        ] * 2
    )

    remote_learner.model = DummyModule()
    assert not module_eq(model, remote_learner.model, remote_dataloader.trace_input)
    model2 = remote_learner.get_model()
    assert module_eq(model, model2, remote_dataloader.trace_input)
