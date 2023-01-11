import torch
import hashlib
import io
from typing import Iterator, TYPE_CHECKING, List, Optional
from dataclasses import dataclass
from ..polars.utils import create_byte_chunk
from ..pb.bastionlab_torch_pb2 import Chunk, Reference, Meta, UpdateTensor
from .utils import DataWrapper, to_torch_meta
from torch.utils.data import Dataset, DataLoader
from ..config import get_client

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from ..polars.remote_polars import RemoteSeries


@dataclass
class RemoteTensor:
    _identifier: str
    _dtype: torch.dtype
    _shape: torch.Size

    @property
    def identifier(self) -> str:
        return self._identifier

    def serialize(self) -> str:
        return f'{{"identifier": "{self.identifier}"}}'

    @staticmethod
    def send_tensor(tensor: torch.Tensor) -> "RemoteTensor":
        client = get_client("torch_client")
        data = DataWrapper([tensor], None)
        ts = torch.jit.script(data)
        buff = io.BytesIO()
        data = torch.jit.save(ts, buff)

        def inner(b) -> Iterator[Chunk]:
            for data in create_byte_chunk(b):
                yield Chunk(
                    data=data, name="", description="", secret=bytes(), meta=bytes()
                )

        res = client.stub.SendTensor(inner(buff.getbuffer()))
        return RemoteTensor._from_reference(res)

    @staticmethod
    def _from_reference(ref: Reference) -> "RemoteTensor":
        dtypes, shape = to_torch_meta(ref.meta)
        return RemoteTensor(ref.identifier, dtypes[0], shape[0])

    def __str__(self) -> str:
        return f"RemoteTensor(identifier={self._identifier}, dtype={self._dtype}, shape={self._shape})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def to(self, dtype: torch.dtype):
        from .utils import tch_kinds

        client = get_client("torch_client")
        res = client.stub.ModifyTensor(
            UpdateTensor(identifier=self.identifier, dtype=tch_kinds[dtype])
        )
        return RemoteTensor._from_reference(res)

    def run_script(self, script: torch.ScriptFunction):
        raise Exception("run_script is unimplemented")

    def fetch_tensor(self) -> torch.Tensor:
        raise Exception("fetch_tensor is unimplemented")


def _tracer(dtypes: List[torch.dtype], shapes: List[torch.Size]):
    return [
        torch.zeros(shape[-1], dtype=dtype)
        if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
        else torch.randn(shape[-1], dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)
    ]


@dataclass
class RemoteDataset:
    inputs: List[RemoteTensor]
    label: RemoteTensor
    name: Optional[str] = "RemoteDataset-" + hashlib.sha256().hexdigest()[:5]
    privacy_limit: Optional[float] = -1.0

    @property
    def trace_input(self):
        dtypes = [input.dtype for input in self.inputs]
        shapes = [input.shape for input in self.inputs]
        return _tracer(dtypes, shapes)

    @property
    def nb_samples(self):
        return self.label.shape[0]

    @staticmethod
    def from_dataset(dataset: Dataset) -> "RemoteDataset":
        data = dataset.__getitem__(0)
        inputs = torch.cat(data[0]).unsqueeze(0)
        labels = torch.tensor(data[1]).unsqueeze(0)

        print("Dataset --> RemoteDataset Transformation")
        for idx in range(1, len(dataset)):
            data = dataset.__getitem__(idx)

            input = torch.cat(data[0]).unsqueeze(0)
            label = torch.tensor(data[1]).unsqueeze(0)

            inputs = torch.cat([inputs, input], 0)
            labels = torch.cat([labels, label])

        print("Transformation Done")
        inputs = RemoteTensor.send_tensor(inputs)
        labels = RemoteTensor.send_tensor(labels.squeeze(0))

        return RemoteDataset([inputs], labels)

    def serialize(self):
        inputs = ",".join([input.serialize() for input in self.inputs])
        return f'{{"inputs": [{inputs}], "label": {self.label.serialize()}, "nb_samples": {self.nb_samples}, "privacy_limit": {self.privacy_limit}}}'

    def __str__(self) -> str:
        return f"RemoteDataset(name={self.name}, privacy_limit={self.privacy_limit}, inputs={str(self.inputs)}, label={str(self.label)})"
