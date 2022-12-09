from typing import Iterator, Tuple, List
import torch
import polars as pl
from ..pb.bastionlab_polars_pb2 import SendChunk
from .policy import Policy

CHUNK_SIZE = 32 * 1024

END_PATTERN = b"[end]"


def create_byte_chunk(data: bytes) -> Iterator[bytes]:
    #: """This method chunks bytes into sub-bytes of len `CHUNK_SIZE = 32KB`.
    #: If `data` is less than 32KB, it returns directly the `data`.
    #: Otherwise, it iteratively yields 32KB until the last chunk of bytes.

    #: Parameters
    #: ----------
    #:     data(bytes): bytes to be chunked.

    #: Returns:
    #:     Iterator[bytes]
    #: """
    sent_bytes = 0
    while sent_bytes < len(data):

        yield bytes(
            data[sent_bytes : sent_bytes + min(CHUNK_SIZE, len(data) - sent_bytes)]
        )

        sent_bytes += min(CHUNK_SIZE, len(data) - sent_bytes)


def serialize_dataframe(
    df: pl.DataFrame, policy: Policy, sanitized_columns: List[str]
) -> Iterator[SendChunk]:
    #:  Converts Polars `DataFrame` to BastionLab `SendChunk`.
    #:  Receives `polars.internals.dataframe.frame.DataFrame` and uses `__getstate__` to convert `DataFrame`
    #:  to `List[bytes]`.

    #: `__getstate__()` internally calls `Series.__getstate__()` which converts
    #: Polars `Series` to `bytes`, and simply adds `b"[END]"` to the end of each `Series.__getstate__()`.

    #: Parameters
    #: ----------
    #: df : polars.internals.dataframe.frame.DataFrame
    #:     Polars DataFrame
    #: policy : Policy
    #:     BastionLab Remote DataFrame policy. This specifies which operations can be performed on
    #:     DataFrames and they specified the _data owner_.
    #: sanitized_columns : List[str]
    #:     This field contains (sensitive) columns in the DataFrame that are to be removed when a Data Scientist
    #:     wishes to fetch a query performed on the DataFrame.

    #: Returns
    #: -------
    #: Iterator[SendChunk]

    #: Example
    #: -------
    #: >>> import polars as pl
    #: >>> data = {"col1": [1, 2, 3, 4]}
    #: >>> df = pl.DataFrame(data)
    #: >>> df
    #: shape (4, 4)
    #:   ┌──────┐
    #:   │ col1 │
    #:   │ ---  │
    #:   │ i64  │
    #:   ╞══════╡
    #:   │ 1    │
    #:   ├╌╌╌╌╌╌┤
    #:   │ 2    │
    #:   ├╌╌╌╌╌╌┤
    #:   │ 3    │
    #:   ├╌╌╌╌╌╌┤
    #:   │ 4    │
    #:   └──────┘

    #: And `df` will be converted to this bytes.
    #: >>> from bastionlab.polars.policy import DEFAULT_POLICY
    #: >>> from bastionlab.polars.utils import serialize_dataframe
    #: >>> next(serialize_dataframe(df, DEFAULT_POLICY, []))
    #: """
    END_PATTERN = b"[end]"
    df_bytes = bytearray()
    for col in df.__getstate__():
        df_bytes += col.__getstate__() + END_PATTERN

    first = True
    for data in create_byte_chunk(df_bytes):
        cols = ",".join([f'"{col}"' for col in sanitized_columns])
        if first:
            first = False
            yield SendChunk(data=data, policy=policy.serialize(), metadata=f"[{cols}]")
        else:
            yield SendChunk(data=data, policy="", metadata="")


def deserialize_dataframe(joined_chunks: bytes) -> pl.DataFrame:
    #: """Converts `bytes` sent from BastionLab `server` to DataFrame.

    #: It notices the `b"[END]"` in the stream of bytes and split the bytes a `List[bytes]`.
    #: Each `bytes` in `List[bytes]` represents a single column in the `polars.DataFrame`.

    #: Each column is converted into a `polars.Series` and then later converted to `polars.DataFrame`.

    #: >>> for i in range(1, len(dfs)):
    #: >>>     out = pl.concat([out, dfs[i]], how="horizontal")

    #: The above section combines `List[polars.DataFrame]` into a single `polars.DataFrame`.

    #: Parameters
    #: ----------
    #: joined_chunks : bytes
    #:     Contains bytes sent from the server.

    #: Returns
    #: -------
    #: polars.internals.dataframe.frame.DataFrame
    #: """
    step = len(END_PATTERN)

    indexes = [0]
    for i in range(0, len(joined_chunks) - step + 1):
        batch = joined_chunks[i : i + step]
        if batch == END_PATTERN:
            indexes.append(i)
    series = []
    for i in range(0, len(indexes) - 2 + 1):
        start = indexes[i]
        end = indexes[i + 1]
        if start == 0:
            start = 0
        else:
            start += 5
        series.append(joined_chunks[start:end])

    dfs = []
    for s in series:
        out = pl.Series()
        out.__setstate__(s)

        dfs.append(pl.DataFrame(out))

    out = dfs[0]
    for i in range(1, len(dfs)):
        out = pl.concat([out, dfs[i]], how="horizontal")
    return out


class ApplyBins(torch.nn.Module):
    #: BastionLab internal class used to serialize user-defined functions (UDF) in TorchScript.
    #: It uses `torch.nn.Module` and stores the `bin_size`, which is the aggregation count of the query.

    def __init__(self, bin_size: int) -> None:
        super().__init__()
        #: The aggregation size of the query.
        self.bin_size = torch.Tensor([bin_size])

    def forward(self, x):
        bins = self.bin_size * torch.ones_like(x)
        return round(x // bins) * bins
