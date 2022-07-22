import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.nn import Module
import io
from pb.remote_torch_pb2 import Chunk
from typing import Any, Iterator, Callable, Tuple, TypeVar, List, Callable
from private_module import trainable_parameters

T = TypeVar('T')

def chunk_bounds(size: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    start = 0
    while start < size:
        yield (start, min(start + chunk_size, size))
        start += chunk_size

def chunks(arr: List[T], chunk_size: int) -> Iterator[List[T]]:
    for a, b in chunk_bounds(len(arr), chunk_size):
        yield arr[a:b]


def tensor_to_bytes(tensor: Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(tensor, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_tensor(bs: bytes) -> Tensor:
    buff = io.BytesIO()
    buff.write(bs)
    buff.seek(0)
    tensor = torch.load(buff)
    return tensor


def stream_artifacts(artifacts: Iterator[T], chunk_size: int, serialization_fn: Callable[[T, io.BytesIO], None] = torch.save) -> Iterator[bytes]:
    eoi = False
    buff = io.BytesIO()
    while not eoi:
        while buff.tell() < chunk_size:
            try:
                tensor = artifacts.__next__()
                header = buff.tell()
                buff.write(b"\xde\xad\xbe\xef")
                serialization_fn(tensor, buff)
                end = buff.tell()
                buff.seek(header)
                buff.write((end - header - 4).to_bytes(4, "little"))
                buff.seek(end)
            except StopIteration:
                eoi = True
                break
        position = buff.tell()
        buff.seek(0)
        if eoi:
            yield buff.read(position)
        else:
            yield buff.read(chunk_size)
            tail = buff.read(position - chunk_size)
            buff.seek(0)
            buff.write(tail)


def unstream_artifacts(stream: Iterator[bytes], deserialization_fn: Callable[[io.BytesIO], T] = torch.load) -> Iterator[T]:
    buff = io.BytesIO()
    init_chunk = stream.__next__()
    size = int.from_bytes(init_chunk[:4], "little")
    buff.write(init_chunk[4:])
    eoi = False

    while not eoi:
        while buff.tell() < size + 4:
            try:
                buff.write(stream.__next__())
            except StopIteration:
                buff.seek(0)
                yield deserialization_fn(buff)
                eoi = True
                break

        if not eoi:
            end = buff.tell()
            buff.seek(size)
            header = buff.read(4)
            size = int.from_bytes(header, "little")
            tail = buff.read(end - size - 4)
            buff.seek(0)
            yield deserialization_fn(buff)
            buff.seek(0)
            buff.write(tail)

def data_chunks_generator(stream: Iterator[bytes], description: str) -> Iterator[Chunk]:
    first = True
    for x in stream:
        if first:
            yield Chunk(data=x, description=description)
        else:
            yield Chunk(data=x, description="")

def serialize_dataset(dataset: Dataset, description: str, chunk_size=1000) -> Iterator[Chunk]:
    return data_chunks_generator(stream_artifacts(iter(dataset), chunk_size), description)

class ArtifactDataset:
    def __init__(self, chunks: Iterator[Chunk]) -> None:
        self.data = list(unstream_artifacts((chunk.data for chunk in chunks))) # type: ignore

    def __len__(self) -> int:
        return len(self.data)
    
    def __get_item__(self, index: int) -> Any:
        return self.data[index]

def serialize_model(model: Module, description: str, chunk_size=1000) -> Iterator[Chunk]:
    ts = torch.jit.script(model)  # type: ignore
    return data_chunks_generator(stream_artifacts(iter([ts]), chunk_size, torch.jit.save), description) # type: ignore

def deserialize_weights_to_model(chunks: Iterator[Chunk], model: Module) -> None:
    tensors = unstream_artifacts((chunk.data for chunk in chunks)) # type: ignore
    for p, t in zip(model.parameters(), tensors):
        p = Parameter(t)

def remote_module(cls: Callable) -> Callable:
    init = cls.__init__

    def new_init(_self, *args, **kwargs):
        init(_self, *args, **kwargs)  # type: ignore
        _self._trainable_parameters = []
        for _, p in trainable_parameters(_self):
            _self._trainable_parameters.append(p)
    cls.__init__ = new_init

    @torch.jit.export  # type: ignore
    def module_trainable_parameters(_self):
        return _self._trainable_parameters
    cls.trainable_parameters = module_trainable_parameters
    return cls
