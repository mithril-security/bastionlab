from typing import Iterator, Tuple, List
import torch
import polars as pl
from ..pb.bastionlab_polars_pb2 import SendChunk
from .policy import Policy

CHUNK_SIZE = 32 * 1024

END_PATTERN = b"[end]"


def create_byte_chunk(data: bytes) -> Tuple[int, Iterator[bytes]]:
    sent_bytes = 0
    while sent_bytes < len(data):

        yield bytes(
            data[sent_bytes : sent_bytes + min(CHUNK_SIZE, len(data) - sent_bytes)]
        )

        sent_bytes += min(CHUNK_SIZE, len(data) - sent_bytes)


def serialize_dataframe(
    df: pl.DataFrame, policy: Policy, sanitized_columns: List[str]
) -> Iterator[SendChunk]:
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
    def __init__(self, bin_size: int) -> None:
        super().__init__()
        self.bin_size = torch.Tensor([bin_size])

    def forward(self, x):
        bins = self.bin_size * torch.ones_like(x)
        return round(x // bins) * bins
