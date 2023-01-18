import torch
import hashlib
import io
from typing import Iterator, TYPE_CHECKING, List, Optional
from dataclasses import dataclass
from ..polars.utils import create_byte_chunk
from .utils import DataWrapper, Chunk
from ..pb.bastionlab_torch_pb2 import UpdateTensor
from ..pb.bastionlab_pb2 import Reference
from torch.utils.data import Dataset, DataLoader
from ..pb.bastionlab_pb2 import Reference, TensorMetaData

if TYPE_CHECKING:
    from ..client import Client


@dataclass
class Metadata:
    _client: "Client"


@dataclass
class RemoteTensor:
    _meta: Metadata
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
        pass
        data = DataWrapper([tensor], None)
        ts = torch.jit.script(data)
        buff = io.BytesIO()
        data = torch.jit.save(ts, buff)

        def inner(b) -> Iterator[Chunk]:
            for data in create_byte_chunk(b):
                yield Chunk(
                    data=data, name="", description="", secret=bytes(), meta=bytes()
                )

        raise Exception(
            "Sending tensors to the BastionLab Torch service is not yet implemented"
        )
        # res = self_meta._torch_client.stub.SendTensor(inner(buff.getbuffer()))
        # return RemoteTensor._from_reference(res)

    @staticmethod
    def _from_reference(ref: Reference, client: "Client") -> "RemoteTensor":
        dtypes, shape = _get_tensor_metadata(ref.meta)
        _meta = Metadata(client)
        return RemoteTensor(_meta, ref.identifier, dtypes[0], shape[0])

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
        res = self._meta._client.torch.stub.ModifyTensor(
            UpdateTensor(identifier=self.identifier, dtype=tch_kinds[dtype])
        )
        return RemoteTensor._from_reference(res, self._meta._client)


torch_dtypes = {
    "Int8": torch.uint8,
    "UInt8": torch.uint8,
    "Int16": torch.int16,
    "Int32": torch.int32,
    "Int64": torch.int64,
    "Half": torch.half,
    "Float": torch.float,
    "Float32": torch.float32,
    "Float64": torch.float64,
    "Double": torch.double,
    "ComplexHalf": torch.complex32,
    "ComplexFloat": torch.complex64,
    "ComplexDouble": torch.complex128,
    "Bool": torch.bool,
    "QInt8": torch.qint8,
    "QInt32": torch.qint32,
    "BFloat16": torch.bfloat16,
}

tch_kinds = {v: k for k, v in torch_dtypes.items()}


def _get_tensor_metadata(meta_bytes: bytes):
    meta = TensorMetaData()
    meta.ParseFromString(meta_bytes)

    return [torch_dtypes[dt] for dt in meta.input_dtype], [
        torch.Size(list(meta.input_shape))
    ]


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
    labels: RemoteTensor
    name: Optional[str] = "RemoteDataset-" + hashlib.sha256().hexdigest()[:5]
    privacy_limit: Optional[float] = -1.0

    @property
    def trace_input(self):
        dtypes = [input.dtype for input in self.inputs]
        shapes = [input.shape for input in self.inputs]
        return _tracer(dtypes, shapes)

    @property
    def nb_samples(self):
        return self.labels.shape[0]

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
        return f'{{"inputs": [{inputs}], "labels": {self.labels.serialize()}, "nb_samples": {self.nb_samples}, "privacy_limit": {self.privacy_limit}}}'

    def __str__(self) -> str:
        return f"RemoteDataset(name={self.name}, privacy_limit={self.privacy_limit}, inputs={str(self.inputs)}, label={str(self.labels)})"
