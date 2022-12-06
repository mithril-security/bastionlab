from typing import List, TYPE_CHECKING, Optional
import grpc
from grpc import StatusCode
import polars as pl
from colorama import Fore
from ..pb.bastionlab_polars_pb2 import (
    ReferenceRequest,
    Query,
    Empty,
)
from ..pb.bastionlab_polars_pb2_grpc import PolarsServiceStub
from ..errors import GRPCException
from .utils import (
    deserialize_dataframe,
    serialize_dataframe,
)
from .policy import Policy, DEFAULT_POLICY


if TYPE_CHECKING:
    from .remote_polars import RemoteLazyFrame, FetchableLazyFrame


class BastionLabPolars:
    def __init__(
        self,
        channel: grpc.Channel,
    ):
        self.stub = PolarsServiceStub(channel)

    def send_df(
        self,
        df: pl.DataFrame,
        policy: Policy = DEFAULT_POLICY,
        sanitized_columns: List[str] = [],
    ) -> "FetchableLazyFrame":
        from .remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(
            lambda: self.stub.SendDataFrame(
                serialize_dataframe(df, policy, sanitized_columns)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(self, ref: List[str]) -> Optional[pl.DataFrame]:
        def inner() -> bytes:
            joined_bytes = b""
            blocked = False

            for b in self.stub.FetchDataFrame(ReferenceRequest(identifier=ref)):
                if blocked:
                    blocked = False
                    print(
                        f"{Fore.GREEN}The query has been accepted by the data owner.{Fore.WHITE}"
                    )

                if b.pending != "":
                    blocked = True
                    print(
                        f"""{Fore.YELLOW}Warning: non privacy-preserving queries necessitate data owner's approval.
Reason: {b.pending}

A notification has been sent to the data owner. The request will be pending until the data owner accepts or denies it or until timeout seconds elapse.{Fore.WHITE}"""
                    )

                if b.warning != "":
                    print(
                        f"""{Fore.YELLOW}Warning: non privacy-preserving query.
Reason: {b.warning}

This incident will be reported to the data owner.{Fore.WHITE}"""
                    )

                joined_bytes += b.data
            return joined_bytes

        try:
            joined_bytes = GRPCException.map_error(inner)
            return deserialize_dataframe(joined_bytes)
        except GRPCException as e:
            if e.code == StatusCode.PERMISSION_DENIED:
                print(
                    f"{Fore.RED}The query has been rejected by the data owner.{Fore.WHITE}"
                )
                return None
            else:
                raise e

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from .remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(
            lambda: self.stub.RunQuery(Query(composite_plan=composite_plan))
        )
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        from .remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(lambda: self.stub.ListDataFrames(Empty()).list)
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        from .remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(
            lambda: self.stub.GetDataFrameHeader(
                ReferenceRequest(identifier=identifier)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)
