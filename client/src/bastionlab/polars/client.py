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
    from ..client import Client


class BastionLabPolars:
    """
    Main BastionLabPolars API class.

    This class contains all the endpoints allowed on the BastionLab server.

    Attributes
    ----------
    stub : bastionlab.pb.bastionlab_polars_pb2_grpc.PolarsServiceStub
        The gRPC service for BastionLab Polars. This define all the API calls for BastionLab Polars.
    """

    def __init__(
        self,
        client: "Client",
    ):
        self.client = client
        self.stub = PolarsServiceStub(client._channel)

    def send_df(
        self,
        df: pl.DataFrame,
        policy: Policy = DEFAULT_POLICY,
        sanitized_columns: List[str] = [],
    ) -> "FetchableLazyFrame":
        """
        This method is used to send `polars.internals.dataframe.frame.DataFrame` to the BastionLab server.

        It readily accepts `polars.internals.dataframe.frame.DataFrame` and also specifies the DataFrame policy and a list of
        sensitive columns.

        Args
        ----
        df : polars.internals.dataframe.frame.DataFrame
            Polars DataFrame
        policy : bastionlab.polars.policy.Policy
            BastionLab Remote DataFrame policy. This specifies which operations can be performed on
            DataFrames and they specified the _data owner_.
        sanitized_columns : List[str]
            This field contains (sensitive) columns in the DataFrame that are to be removed when a Data Scientist
            wishes to fetch a query performed on the DataFrame.

        Returns
        -------
        bastionlab.polars.remote_polars.FetchableLazyFrame

        Example
        -------
        Import the necessary packages
        >>> import polars as pl
        >>> from bastionlab import Connection
        >>> from bastionlab.polars.policy import Policy, Aggregation, Log


        We create the DataFrame locally
        >>> data = {"col1": [1, 2, 3, 4]}
        >>> df = pl.DataFrame(data)

        We create a connection to the BastionLab server running `locally`.
        >>> connection = Connection("localhost", identity=data_owner)

        Here, we create a sample `Policy`.
        >>> policy = Policy(safe_zone=Aggregation(min_agg_size=2), unsafe_handling=Log())

        We send the DataFrame to the server.
        >>> connection.client.polars.send_df(df, policy=policy, sanitized_columns=["Name"])
        """
        from .remote_polars import FetchableLazyFrame

        self.client.refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.SendDataFrame(
                serialize_dataframe(df, policy, sanitized_columns)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(self, ref: str) -> Optional[pl.DataFrame]:
        #: Fetches the specified `pl.DataFrame` from the BastionLab server
        #: with the provided reference identifier.

        #: Parameters
        #: ----------
        #: ref : str
        #:     A unique identifier for the Remote DataFrame.

        #: Returns
        #: -------
        #: Optional[pl.DataFrame]
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

        self.client.refresh_session_if_needed()

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

        #: Executes a Composite Plan on the BastionLab server.
        #: A composite plan is BastionLab's internal instruction set.

        #: Parameters
        #: ----------
        #: composite_plan : str
        #:     Serialized instructions to be executed on BastionLab server.

        #: Returns
        #: -------
        #: bastionlab.polars.remote_polars.FetchableLazyFrame

        from .remote_polars import FetchableLazyFrame

        self.client.refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.RunQuery(Query(composite_plan=composite_plan))
        )
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        """
        Enlists all the DataFrames available on the BastionLab server.

        Returns
        -------
        List[bastionlab.polars.remote_polars.FetchableLazyFrame]

        """
        from .remote_polars import FetchableLazyFrame

        self.client.refresh_session_if_needed()

        res = GRPCException.map_error(lambda: self.stub.ListDataFrames(Empty()).list)
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        """
        Returns a `bastionlab.polars.remote_polars.FetchableLazyFrame` from an BastionLab DataFrame identifier.

        Args
        ----
        identifier : str
            A unique identifier for the Remote DataFrame.

        Returns
        -------
        bastionlab.polars.remote_polars.FetchableLazyFrame
        """
        from .remote_polars import FetchableLazyFrame

        self.client.refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.GetDataFrameHeader(
                ReferenceRequest(identifier=identifier)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)

    def persist_df(self, identifier: str):
        """
        Saves a Dataframe on the server from a BastionLab DataFrame identifier.

        Args
        ----
        identifier : str
            A unique identifier for the Remote DataFrame.

        Returns
        -------
        Nothing
        """
        res = GRPCException.map_error(
            lambda: self.stub.PersistDataFrame(ReferenceRequest(identifier=identifier))
        )
