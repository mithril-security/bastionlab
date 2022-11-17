from dataclasses import dataclass
from typing import Any, List, TYPE_CHECKING, Optional
import grpc
from bastionlab.pb.bastionlab_pb2 import (
    ReferenceRequest,
    Query,
    Empty,
)
from bastionlab.pb.bastionlab_pb2_grpc import BastionLabStub
import polars as pl
from colorama import Fore

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
        blocked = False
        for b in self.stub.FetchDataFrame(ReferenceRequest(identifier=ref)):
            if blocked:
                blocked = False
                print(f"{Fore.GREEN}The query has been approved by the data owner.{Fore.WHITE}")
            if b.pending != "":
                print(b.pending)
                blocked = True
                print(f"""{Fore.YELLOW}Warning: non privacy-preserving queries necessitate data owner's approval.
Reason: {b.pending}
A notification has been sent to the data owner. The request will be pending until the data owners accepts or denies it or until timeout seconds elapse.{Fore.WHITE}""")
            joined_bytes += b.data

        return deserialize_dataframe(joined_bytes)

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.RunQuery(Query(composite_plan=composite_plan))
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.ListDataFrames(Empty()).list
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.GetDataFrameHeader(ReferenceRequest(identifier=identifier))
        return FetchableLazyFrame._from_reference(self, res)


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
