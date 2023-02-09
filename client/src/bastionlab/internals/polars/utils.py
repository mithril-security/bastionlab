from typing import Iterator, Tuple, List
import torch
import polars as pl
import io
from bastionlab.pb.bastionlab_polars_pb2 import SendChunk
from .policy import Policy
from serde.json import to_json

CHUNK_SIZE = 32 * 1024

# TODO PERF: Do a PR on polars/pypolars to add the streaming IPC (apache flight) format to the python interface
# right now, there is only the file format which requires random access
# which means, we have to do a full copy to a buffer and we cannot parse it as we go


def serialize_dataframe(
    df: pl.DataFrame, policy: Policy, sanitized_columns: List[str]
) -> Iterator[SendChunk]:
    """Converts Polars `DataFrame` to BastionLab `SendChunk` protobuf message.
    This currently uses the Apache IPC format.
    Args:
        df : pl.DataFrame
            Polars DataFrame
        policy : Policy
            BastionLab Remote DataFrame policy. This specifies which operations can be performed on
            DataFrames and they specified the data owner.
        sanitized_columns : List[str]
            This field contains the sensitive columns in the DataFrame that will be removed when a Data Scientist
            wishes to fetch a query performed on the DataFrame.
    Returns:
        Iterator[SendChunk]
    """
    buf = io.BytesIO()

    df.write_ipc(buf)

    buf.seek(0)
    max = len(buf.getvalue())
    first = True
    while buf.tell() < max:
        data = buf.read(CHUNK_SIZE)

        if first:
            chunk = SendChunk(
                data=data,
                policy=to_json(policy),
                sanitized_columns=sanitized_columns,
            )
            first = False
        else:
            chunk = SendChunk(data=data)

        yield chunk


def deserialize_dataframe(chunks: Iterator[bytes]) -> pl.DataFrame:
    """Converts chunks of `bytes` sent from BastionLab server to DataFrame.
    This currently uses the Apache IPC format.
    Args:
        chunks : Iterator[bytes]
            Iterator of bytes sent from the server.
    Returns:
        pl.DataFrame
    """
    buf = io.BytesIO()

    for el in chunks:
        buf.write(el)

    buf.seek(0)
    df = pl.read_ipc(buf)

    return df


class ApplyBins(torch.nn.Module):
    """BastionLab internal class used to serialize user-defined functions (UDF) in TorchScript.
    It uses `torch.nn.Module` and stores the `bin_size`, which is the aggregation count of the query.
    """

    def __init__(self, bin_size: int) -> None:
        super().__init__()
        #: The aggregation size of the query.
        self.bin_size = torch.Tensor([bin_size])

    def forward(self, x):
        bins = self.bin_size * torch.ones_like(x)
        return round(x // bins) * bins


class Palettes:
    dict = {
        "standard": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "light": ["#add8e6", "#ffb6c1", "#cbc3e3", "#ff8520", "#7ded7f", "#92f7e4"],
        "mithril": ["#f0ba2d", "#0b2440", "#030e1a", "#ffffff", "#F74C00"],
        "ocean": ["#006A94", "#2999BC", "#3EBDC8", "#69D1CB", "#83DEF1", "#01BFFF"],
    }


class ApplyAbs(torch.nn.Module):
    """BastionLab internal class used to serialize user-defined functions (UDF) in TorchScript.
    It uses `torch.nn.Module` and applies abs() to the input value.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.abs(x)
