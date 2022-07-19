import torch
from torch.utils.data import Dataset
from torch.nn import Module
import io
from remote_torch_pb2 import Chunk
from typing import Iterator, Callable
from private_module import trainable_parameters

def chunk_bounds(size, chunk_size):
    start = 0
    while start < size:
        yield (start, min(start + chunk_size, size))
        start += chunk_size

def chunks(arr, chunk_size):
    for a, b in chunk_bounds(len(arr), chunk_size):
        yield arr[a:b]

def tensor_to_bytes(tensor):
    buff = io.BytesIO()
    torch.save(tensor, buff)
    buff.seek(0)
    return buff.read()

def bytes_to_tensor(bs):
    buff = io.BytesIO()
    buff.write(bs)
    buff.seek(0)
    tensor = torch.load(buff)
    return tensor

def stream_artifacts(artifacts, chunk_size, serialization_fn=torch.save):
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

def unstream_artifacts(stream, deserialization_fn=torch.load):
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

def serialize_dataset(dataset: Dataset, chunk_size=1000) -> Iterator[Chunk]:
    return (Chunk(data=x) for x in stream_artifacts(iter(dataset), chunk_size))

def serialize_model(model: Module, chunk_size=1000) -> Iterator[Chunk]:
    ts = torch.jit.script(model) # type: ignore
    return (Chunk(data=x) for x in stream_artifacts(iter([ts]), chunk_size, torch.jit.save)) # type: ignore

def remote_module(cls: Callable) -> Callable:
    init = cls.__init__
    def new_init(_self, *args, **kwargs):
        init(_self, *args, **kwargs) # type: ignore
        _self._trainable_parameters = []
        for _, p in trainable_parameters(_self):
            _self._trainable_parameters.append(p)
    cls.__init__ = new_init

    @torch.jit.export # type: ignore
    def module_trainable_parameters(_self):
        return _self._trainable_parameters
    cls.trainable_parameters = module_trainable_parameters
    return cls
