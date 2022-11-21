from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING, Optional
from hashlib import sha256
import grpc
from bastionlab.pb.bastionlab_pb2 import (
    ReferenceRequest,
    ReferenceResponse,
    ClientInfo,
    Query,
    Empty,
)
from bastionlab.version import __version__ as app_version
from bastionlab.pb.bastionlab_pb2_grpc import BastionLabStub
import platform
import socket
import getpass
import polars as pl

from bastionlab.utils import (
    deserialize_dataframe,
    serialize_dataframe,
    send_clientinfo,
)

if TYPE_CHECKING:
    from bastionlab.remote_polars import RemoteLazyFrame, FetchableLazyFrame


class Client:
    def __init__(self, stub: BastionLabStub):
        self.stub = stub
        uname = platform.uname()
        self.client_info = ClientInfo(
            uid=sha256((socket.gethostname() + "-" + getpass.getuser()).encode("utf-8"))
            .digest()
            .hex(),
            platform_name=uname.system,
            platform_arch=uname.machine,
            platform_version=uname.version,
            platform_release=uname.release,
            user_agent="bastionlab_python",
            user_agent_version=app_version,
        )

    def send_df(self, df: pl.DataFrame) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.SendDataFrame(serialize_dataframe(df))
        res_client_info = self.stub.SendDataFrame(send_clientinfo(self.client_info))
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(self, ref: List[str]) -> pl.DataFrame:
        joined_bytes = b""
        for b in self.stub.FetchDataFrame(ReferenceRequest(identifier=ref, client_info=self.client_info)):
            joined_bytes += b.data

        return deserialize_dataframe(joined_bytes)

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.RunQuery(Query(composite_plan=composite_plan, client_info=self.client_info))
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
