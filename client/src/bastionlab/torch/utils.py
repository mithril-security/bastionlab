import io
from typing import Callable, Iterator, List, Tuple, TypeVar, Optional, Any
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore [import]
from ..pb.bastionlab_torch_pb2 import Chunk, Reference  # type: ignore [import]

T = TypeVar("T")
U = TypeVar("U")
SIZE_LEN = 8


class DataWrapper(Module):
    """Wrapps data in a (dummy) module to be able to retrieve them through libtorch on the server.

    Args:
        columns: Tensors that represent the clolumns of the dataset (a column contains the values
        for a given input for all samples).
        labels: A tensor containing the labels of all inputs.
        privacy_limit: Maximum privacy budget that can be expanded on these data.
    """

    def __init__(
        self,
        columns: List[Tensor],
        labels: Optional[Tensor],
        privacy_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        for i, column in enumerate(columns):
            self.__setattr__(f"samples_{i}", Parameter(column, requires_grad=False))
        if labels is not None:
            self.labels = Parameter(labels, requires_grad=False)
        self.privacy_limit = Parameter(
            torch.tensor([privacy_limit if privacy_limit is not None else -1.0])
        )


class TensorDataset(Dataset):
    """A simple dataset compliant with Torch's `Dataset` build upon
    tensors representing columns and labels.

    Args:
        columns: Tensors that represent the clolumns of the dataset (a column contains the values
        for a given input for all samples).
        labels: A tensor containing the labels of all inputs.
    """

    def __init__(self, columns: List[Tensor], labels: Optional[Tensor]) -> None:
        super().__init__()
        self.columns = columns
        self.labels = labels

    def __len__(self) -> int:
        return len(self.columns[0])

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], Optional[Tensor]]:
        return (
            [column[idx] for column in self.columns],
            self.labels[idx] if self.labels is not None else None,
        )


def data_from_wrapper(wrapper: DataWrapper) -> Tuple[List[Tensor], Optional[Tensor]]:
    """Converts a data wrapper into a list of columns and labels."""
    opt_columns: List[Optional[Tensor]] = []
    labels: Optional[Tensor] = None
    for name, param in wrapper.named_parameters():
        if name == "labels":
            labels = param
        elif name.startswith("samples_"):
            idx = int(name[8:])
            print(idx)
            if len(opt_columns) <= idx:
                opt_columns += [None] * (idx + 1 - len(opt_columns))
            opt_columns[idx] = param
        else:
            if name not in ["privacy_limit"]:
                raise Exception(f"Unknown field {name} in data wrapper")
    if len(opt_columns) == 0:
        raise Exception(f"Data wrapper must contain at least one column.")
    if any([x is None for x in opt_columns]):
        raise Exception(f"Missing column in data wrapper.")
    columns: List[Tensor] = [x for x in opt_columns if x is not None]

    return (columns, labels)


def dataset_from_chunks(chunks: Iterator[Chunk]) -> TensorDataset:
    """Builds a `TensorDataset` from a chunks iterator (returned by the underlying gRPC protocol)."""
    wrappers = unstream_artifacts(
        (chunk.data for chunk in chunks), deserialization_fn=torch.jit.load
    )

    data = [data_from_wrapper(wrapper) for wrapper in wrappers]

    columns = [torch.cat([x[i] for x, _ in data]) for i in range(len(data[0][0]))]
    labels = (
        torch.cat([y for _, y in data if y is not None])
        if data[0][1] is not None
        else None
    )

    return TensorDataset(columns, labels)


def id(l: List[T], _: Optional[U] = None) -> U:
    """Identity functions. Here for type checking."""
    res: U = l  # type: ignore [assignment]
    return res


def chunks(
    it: Iterator[T], chunk_size: int, cat_fn: Callable[[List[T]], U] = id
) -> Iterator[U]:
    """Groups elements of an iterator by chunks.

    Args:
        it: input iterator.
        chunk_size: size of the chunks.
        cat_fn: aggregation function called on every chunk.
    """
    chunk: List[T] = []
    for x in it:
        if len(chunk) == chunk_size:
            yield cat_fn(chunk)
            chunk = [x]
        else:
            chunk.append(x)
    if len(chunk) > 0:
        yield cat_fn(chunk)


def serialize_batch(
    privacy_limit: Optional[float] = None,
) -> Callable[[Tuple[List[Tensor], Tensor], io.BytesIO], None]:
    """Serializes a batch of data by wrapping it in a `DataWrapper` module, using torch.jit utility functions and
    writes the output to the given buffer.
    """

    def inner(data: Tuple[List[Tensor], Tensor], buff: io.BytesIO) -> None:
        torch.jit.save(torch.jit.script(DataWrapper(*data, privacy_limit)), buff)

    return inner


def stream_artifacts(
    artifacts: Iterator[T],
    chunk_size: int,
    serialization_fn: Callable[[T, io.BytesIO], None] = torch.save,
) -> Iterator[Tuple[int, bytes]]:
    """Converts an iterator of objects into an iterator of bytes chunks.

    Args:
        artifacts: Iterator whose objects will be converted.
        chunk_size: Size of the bytes chunks.
        serialization_fn: Function used to convert an object into bytes and write these bytes to a buffer.
    """
    eoi = False
    buff = io.BytesIO()
    while not eoi:
        while buff.tell() < chunk_size:
            try:
                artifact = artifacts.__next__()
                header = buff.tell()
                buff.write(b"\xde\xad\xbe\xef\xde\xad\xbe\xef")
                serialization_fn(artifact, buff)
                end = buff.tell()
                buff.seek(header)
                buff_len = (end - header - SIZE_LEN).to_bytes(
                    SIZE_LEN, byteorder="little"
                )
                buff.write(buff_len)
                buff.seek(end)
            except StopIteration:
                eoi = True
                break
        position = buff.tell()
        buff.seek(0)
        if eoi:
            yield (position, buff.read(position))
        else:
            yield (position, buff.read(chunk_size))
            tail = buff.read(position - chunk_size)
            buff.seek(0)
            buff.write(tail)


def unstream_artifacts(
    stream: Iterator[bytes], deserialization_fn: Callable[[io.BytesIO], T] = torch.load
) -> Iterator[T]:
    """Converts an iterator of bytes chunks into an iterator of objects.

    Args:
        stream: Iterator of bytes chunks.
    """
    buff = io.BytesIO()
    init_chunk = stream.__next__()
    size = int.from_bytes(init_chunk[:8], "little")
    buff.write(init_chunk[8:])
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
            header = buff.read(SIZE_LEN)
            size = int.from_bytes(header, "little")
            tail = buff.read(end - size - SIZE_LEN)
            buff.seek(0)
            yield deserialization_fn(buff)
            buff.seek(0)
            buff.write(tail)


def data_chunks_generator(
    stream: Iterator[Tuple[int, bytes]],
    name: str,
    description: str,
    meta: bytes,
    progress: bool = False,
) -> Iterator[Chunk]:
    """Converts an iterator of bytes chunks into an iterator of BastionAI gRPC protocol `Chunk` messages.

    Args:
        stream: Iterator of bytes chunks.
        name: A name for the objects being sent.
        description: Description of the objects being sent.
    """
    first = True
    last_estimate = 0
    t = None
    for estimate, x in stream:
        # print(estimate)
        # print(f"{chunk_size} > {estimate}")

        if progress and estimate > last_estimate:
            if t is not None:
                t.total = t.n + estimate
                t.refresh()
            else:
                t = tqdm(
                    total=estimate,
                    unit="B",
                    unit_scale=True,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                )
                t.set_description(f"Sending {name}")

        if first:
            first = False
            yield Chunk(data=x, name=name, description=description, meta=meta)
        else:
            yield Chunk(data=x, name=name, description="", meta=bytes())

        if progress and t is not None:
            t.update(len(x))
        last_estimate = estimate


def make_batch(data: List[Tuple[List[Tensor], Tensor]]) -> Tuple[List[Tensor], Tensor]:
    """Aggregates a group of lists of column tensors and label tensors."""
    return (
        [torch.stack([x[0][i] for x in data]) for i in range(len(data[0][0]))],
        torch.stack([x[1] for x in data])
        if type(data[0][1]) == torch.Tensor
        else torch.tensor([x[1] for x in data]),
    )


def serialize_dataset(
    dataset: Dataset,
    name: str,
    description: str,
    privacy_limit: Optional[float] = None,
    chunk_size: int = 100_000_000,
    batch_size: int = 1024,
    train_dataset: Optional[Reference] = None,
    progress: bool = False,
) -> Iterator[Chunk]:
    """Coverts a dataset into an iterator of bytes chunks.

    The dataset is processed one batch at a time. Each batch is wrapped in a `DataModuleWrapper`
    and serialized using `torch.jit` utility functions.

    Args:
        dataset: Dataset to be serialized.
        name: Name of the dataset on the server.
        description: Description of the dataset.
        privacy_limit: Maximum privacy budget that can be spent on this dataset.
        chunk_size: size of the bytes chunks sent over gRPC.
        batch_size: size of the batches (in number of samples) during the serialization step.
        train_dataset: metadata, True means this dataset is suited for training, False that it should be used for testing/validating only
    """
    return data_chunks_generator(
        stream_artifacts(
            chunks(iter(dataset), batch_size, cat_fn=make_batch),
            chunk_size,
            serialization_fn=serialize_batch(privacy_limit),
        ),
        name=name,
        description=description,
        meta=bulk_serialize(
            {
                "input_shape": [input.size() for input in dataset[0][0]],
                "input_dtype": [input.dtype for input in dataset[0][0]],
                "nb_samples": len(dataset),  # type: ignore [arg-type]
                "privacy_limit": privacy_limit,
                "train_dataset": train_dataset,
            }
        ),
        progress=progress,
    )


def serialize_model(
    model: Module,
    name: str,
    description: str,
    chunk_size: int = 100_000_000,
    progress: bool = False,
) -> Iterator[Chunk]:
    """Coverts a model into an iterator of bytes chunks.

    The model is scripted using `torch.jit.script` and serialiazed as a whole with `torch.jit.save`. The resulting
    bytes are sent over gRPC in a chunked manner.

    Args:
        model: Model to be serialized.
        name: Name of the model on the server.
        description: Description of the model.
        chunk_size: size of the bytes chunks sent over gRPC.
    """
    ts = torch.jit.script(model)
    return data_chunks_generator(
        stream_artifacts(iter([ts]), chunk_size, torch.jit.save),
        name=name,
        description=description,
        meta=b"",
        progress=progress,
    )


def deserialize_weights_to_model(model: Module, chunks: Iterator[Chunk]) -> None:
    """Deserializes weights from an iterator of BastionAI gRPC protocol `Chunks` writes
    them to the passed model.
    """
    wrapper = list(
        unstream_artifacts(
            (chunk.data for chunk in chunks), deserialization_fn=torch.jit.load
        )
    )[0]
    for name, value in wrapper.named_parameters():
        param = model
        parent = None
        segments = name.split("_")
        name_buf = []
        for segment in segments:
            name_buf.append(segment)
            name = "_".join(name_buf)
            if hasattr(param, name):
                parent = param
                param = param.__getattr__(name)  # type: ignore [assignment]
                name_buf = []

        parent.__setattr__(name, torch.nn.Parameter(value))


class MultipleOutputWrapper(Module):
    """Utility wrapper to select one output of a model with multiple outputs.

    Args:
        module: A model with more than one outputs.
        output: Index of the output to retain.
    """

    def __init__(self, module: Module, output: int = 0) -> None:
        super().__init__()
        self.inner = module
        self.output = output

    def forward(self, *args, **kwargs) -> Tensor:
        output = self.inner.forward(*args, **kwargs)
        return output[self.output]


def bulk_serialize(obj: Any) -> bytes:
    buff = io.BytesIO()
    torch.save(obj, buff)
    buff.seek(0)
    return buff.read()


def bulk_deserialize(b: bytes) -> Any:
    buff = io.BytesIO()
    buff.write(b)
    buff.seek(0)
    return torch.load(buff)
