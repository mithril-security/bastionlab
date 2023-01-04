import torch
from typing import Iterator, TYPE_CHECKING, Tuple, List
import io
from dataclasses import dataclass
from ..polars.utils import create_byte_chunk
from ..pb.bastionlab_torch_pb2 import Chunk, Reference, Meta
from .utils import DataWrapper, to_torch_meta

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from ..polars.remote_polars import RemoteSeries


def get_client():
    from ..config import CONFIG

    if CONFIG["torch_client"] == None:
        raise Exception("BastionLab Torch client is not initialized.")

    return CONFIG["torch_client"]


@dataclass
class RemoteTensor:
    _identifier: str
    _dtype: torch.dtype
    _shape: torch.Size

    @property
    def identifier(self) -> str:
        return self._identifier

    @staticmethod
    def send_tensor(tensor: torch.Tensor) -> "RemoteTensor":
        client = get_client()
        data = DataWrapper([tensor], None)
        ts = torch.jit.script(data)
        buff = io.BytesIO()
        data = torch.jit.save(ts, buff)

        def inner(b) -> Iterator[Chunk]:
            for data in create_byte_chunk(b):
                yield Chunk(
                    data=data, name="", description="", secret=bytes(), meta=bytes()
                )

        res = client.stub.SendTensor(inner(buff.getvalue()))
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

    def run_script(self, script: torch.ScriptFunction):
        pass

    def fetch_tensor(self) -> torch.Tensor:
        pass


@dataclass
class RemoteDataset:
    inputs: List[RemoteTensor]
    label: RemoteTensor

    # def
