from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING, Optional
import grpc
from bastionlab.pb.bastionlab_pb2 import ReferenceRequest, TrainingRequest, Query, Empty
from bastionlab.pb.bastionlab_pb2_grpc import BastionLabStub
import polars as pl

from bastionlab.utils import (
    deserialize_dataframe,
    serialize_dataframe,
)

if TYPE_CHECKING:
    from bastionlab.remote_polars import RemoteLazyFrame, FetchableLazyFrame


class Client:
    def __init__(self, stub: BastionLabStub):
        self.stub = stub

    def send_df(self, df: pl.DataFrame) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.SendDataFrame(serialize_dataframe(df))
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(self, ref: List[str]) -> pl.DataFrame:
        joined_bytes = b""
        for b in self.stub.FetchDataFrame(ReferenceRequest(identifier=ref)):
            joined_bytes += b.data

        return deserialize_dataframe(joined_bytes)

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.RunQuery(
            Query(composite_plan=composite_plan)
        )
        return FetchableLazyFrame._from_reference(self, res)

    def available_datasets(self) -> List[str]:
        def create_schema(obj: str) -> Dict:
            items = obj.split(', ')
            return {"field": items[0].split(": ")[1], "data_type": items[1].split(": ")[1]}

        def remove_empty(s):
            return list(filter(lambda a: len(a) > 0 and "Schema" not in a, s))

        def create_dataset(identifier, schema: str):
            return {"identifier": identifier, "schema": [create_schema(x) for x in remove_empty(schema.split("\n"))]}

        res = self.stub.AvailableDatasets(Empty()).list
        return [create_dataset(x.identifier, x.header) for x in res]

    def train(self, records: "FetchableLazyFrame", target: "FetchableLazyFrame", ratio: float, trainer: str):
        res = self.stub.Train(TrainingRequest(
            records=records.identifier,
            target=target.identifier,
            ratio=ratio,
            trainer=trainer))
        return res


@dataclass
class Connection:
    host: str
    port: int
    channel: Any = None
    _client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if self._client is not None:
            return self._client
        else:
            return self.__enter__()

    def close(self):
        if self._client is not None:
            self.__exit__(None, None, None)

    def __enter__(self) -> Client:
        server_target = f"{self.host}:{self.port}"
        self.channel = grpc.insecure_channel(server_target)
        self._client = Client(BastionLabStub(self.channel))
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
