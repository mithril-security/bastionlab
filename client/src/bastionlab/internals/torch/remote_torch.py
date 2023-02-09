import torch
from typing import TYPE_CHECKING, List, Optional, Any
from dataclasses import dataclass
from .utils import send_tensor
from bastionlab.pb.bastionlab_torch_pb2 import UpdateTensor, RemoteDatasetReference
from bastionlab.pb.bastionlab_pb2 import Reference
from torch.utils.data import Dataset
from bastionlab.pb.bastionlab_pb2 import Reference, TensorMetaData

if TYPE_CHECKING:
    from .client import BastionLabTorch


@dataclass
class RemoteTensor:
    """
    BastionLab reference to a PyTorch (tch) Tensor on the server.

    It also stores a few basic information about the tensor (`dtype`, `shape`).

    You can also change the dtype of the tensor through an API call
    """

    _client: "BastionLabTorch"
    _identifier: str
    _dtype: torch.dtype
    _shape: torch.Size

    @property
    def identifier(self) -> str:
        return self._identifier

    def _serialize(self) -> Any:
        return {"identifier": {self.identifier}}

    @staticmethod
    def _send_tensor(client: "BastionLabTorch", tensor: torch.Tensor) -> "RemoteTensor":
        res = client.stub.SendTensor(send_tensor(tensor))
        dtype, shape = _get_tensor_metadata(res.meta)
        return RemoteTensor(client, res.identifier, *dtype, *shape)

    @staticmethod
    def _from_reference(ref: Reference, client: "BastionLabTorch") -> "RemoteTensor":
        dtypes, shape = _get_tensor_metadata(ref.meta)
        return RemoteTensor(client, ref.identifier, dtypes[0], shape[0])

    def __str__(self) -> str:
        return f"RemoteTensor(identifier={self._identifier}, dtype={self._dtype}, shape={self._shape})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def dtype(self) -> torch.dtype:
        """Returns the torch dtype of the corresponding tensor"""
        return self._dtype

    @property
    def shape(self):
        """Returns the torch Size of the corresponding tensor"""
        return self._shape

    def to(self, dtype: torch.dtype):
        """
        Performs Tensor dtype conversion.

        Args:
            dtype: torch.dtype
                The resulting torch.dtype
        """
        res = self._client.torch.stub.ModifyTensor(
            UpdateTensor(identifier=self.identifier, dtype=tch_kinds[dtype])
        )
        return RemoteTensor._from_reference(res, self._client)


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


def _make_id_reference(id: str) -> Reference:
    return Reference(identifier=id, description="", name="", meta=bytes())


@dataclass
class RemoteDataset:
    inputs: List[RemoteTensor]
    labels: RemoteTensor
    name: Optional[str] = "RemoteDataset"
    description: Optional[str] = "RemoteDataset"
    privacy_limit: Optional[float] = -1.0
    identifier: Optional[str] = ""

    @property
    def _trace_input(self):
        dtypes = [input.dtype for input in self.inputs]
        shapes = [input.shape for input in self.inputs]
        return _tracer(dtypes, shapes)

    @property
    def nb_samples(self) -> int:
        """Returns the number of samples in the RemoteDataset"""
        return self.labels.shape[0]

    @property
    def input_dtype(self) -> torch.dtype:
        """Returns the input dtype of the tensors stored"""
        return self.labels.dtype

    @staticmethod
    def _from_remote_tensors(
        client: "BastionLabTorch",
        inputs: List["RemoteTensor"],
        labels: "RemoteTensor",
        **kwargs,
    ) -> "RemoteDataset":
        name = kwargs.get("name", "")
        description = kwargs.get("description", "")
        privacy_limit = kwargs.get("privacy_limit", -1.0)
        inputs = [_make_id_reference(input.identifier) for input in inputs]
        labels = Reference(
            identifier=labels.identifier,
            description=description,
            name=name,
            meta=bytes(f'{{"privacy_limit": {privacy_limit}}}', encoding="ascii"),
        )

        res = client.stub.ConvToDataset(
            RemoteDatasetReference(
                identifier="",
                inputs=inputs,
                labels=labels,
            )
        )

        inputs = [RemoteTensor._from_reference(ref, client) for ref in res.inputs]
        labels = RemoteTensor._from_reference(res.labels, client)

        return RemoteDataset(
            inputs,
            labels,
            name=name,
            description=description,
            privacy_limit=privacy_limit,
            identifier=res.identifier,
        )

    @staticmethod
    def _from_dataset(
        client: "BastionLabTorch", dataset: Dataset, *args, **kwargs
    ) -> "RemoteDataset":
        res: RemoteDatasetReference = client.send_dataset(dataset, *args, **kwargs)
        inputs = [RemoteTensor._from_reference(ref, client) for ref in res.inputs]
        labels = RemoteTensor._from_reference(res.labels, client)

        name = kwargs.get("name")
        description = kwargs.get("description")
        privacy_limit = kwargs.get("privacy_limit")

        return RemoteDataset(
            inputs,
            labels,
            name=name,
            description=description,
            privacy_limit=-1.0 if not privacy_limit else privacy_limit,
            identifier=res.identifier,
        )

    def _serialize(self) -> Any:
        return {
            "inputs": [input._serialize() for input in self.inputs],
            "labels": self.labels._serialize(),
            "nb_samples": self.nb_samples,
            "privacy_limit": self.privacy_limit,
            "identifier": self.identifier,
        }

    def __str__(self) -> str:
        return f"RemoteDataset(identifier={self.identifier}, name={self.name}, privacy_limit={self.privacy_limit}, inputs={str(self.inputs)}, label={str(self.labels)})"
