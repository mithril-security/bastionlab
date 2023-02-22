from typing import Iterator, List, Union, TYPE_CHECKING
import polars as pl
import io
from ..pb.bastionlab_polars_pb2 import SendChunk
from .policy import Policy
from serde.json import to_json
from dataclasses import dataclass
from torch.jit import ScriptFunction
import base64
import json
from serde import serde, InternalTagging, field
from serde.json import to_json

if TYPE_CHECKING:
    from .client import BastionLabPolars

CHUNK_SIZE = 32 * 1024

# TODO PERF: Do a PR on polars/pypolars to add the streaming IPC (apache flight) format to the python interface
# right now, there is only the file format which requires random access
# which means, we have to do a full copy to a buffer and we cannot parse it as we go


class CompositePlanSegment:
    """
    Composite plan segment class which handles segment plans that have not been implemented
    """


@dataclass
@serde
class EntryPointPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for new entry points
    """

    identifier: str


@dataclass
@serde
class PolarsPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for Polars queries
    """

    # HACK: when getting using the schema attribute, polars returns
    #  the proper error messages (polars.NotFoundError etc) when it is invalid.
    #  This is not the case for write_json(), which returns a confusing error
    #  message. So, we get the schema beforehand :)
    plan: pl.LazyFrame = field(
        serializer=lambda val: val.schema and json.loads(val.write_json()),
        deserializer=lambda _: None,
    )


@dataclass
@serde
class UdfPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for user defined functions
    """

    columns: List[str]
    udf: ScriptFunction = field(
        serializer=lambda val: base64.b64encode(val.save_to_buffer()).decode("ascii"),
        deserializer=lambda _: None,
    )


@dataclass
@serde
class StackPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for vstack function
    """


@dataclass
@serde
class RowCountSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for with_row_count function
    """

    row: str


@dataclass
@serde(tagging=InternalTagging("type"))
class PlanSegments:
    segments: List[
        Union[
            PolarsPlanSegment,
            UdfPlanSegment,
            EntryPointPlanSegment,
            StackPlanSegment,
            RowCountSegment,
        ]
    ]


@dataclass
class UdfTransformerPlanSegment(CompositePlanSegment):
    """
    Accepts a UDF for row-wise DataFrame transformation.
    """

    _name: str
    _columns: List[str]

    def serialize(self) -> str:
        pass


def serialize_dataframe(
    df: pl.DataFrame, policy: Policy, sanitized_columns: List[str]
) -> Iterator[SendChunk]:
    """Converts Polars `DataFrame` to BastionLab `SendChunk` protobuf message.
    This currently uses the Apache IPC format.
    Args:
        df : polars.internals.dataframe.frame.DataFrame
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
        polars.internals.dataframe.frame.DataFrame
    """
    buf = io.BytesIO()

    for el in chunks:
        buf.write(el)

    buf.seek(0)
    df = pl.read_ipc(buf)

    return df


@dataclass
class Metadata:
    """
    A class containing metadata related to your dataframe
    """

    _polars_client: "BastionLabPolars"
    _prev_segments: List[CompositePlanSegment] = field(default_factory=list)
