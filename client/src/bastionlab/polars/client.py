from typing import List, TYPE_CHECKING, Optional, Iterator
from grpc import StatusCode
import polars as pl
from colorama import Fore
from ..pb.bastionlab_polars_pb2 import ReferenceRequest, Empty, Query
from ..pb.bastionlab_polars_pb2_grpc import PolarsServiceStub
from ..pb.bastionlab_pb2 import Reference
from ..errors import GRPCException
from ._utils import deserialize_dataframe, serialize_dataframe
from .policy import Policy, DEFAULT_POLICY


if TYPE_CHECKING:
    import bastionlab.polars.frame
    from .frame import FetchableLazyFrame, RemoteArray
    from ..client import Client


__pdoc__ = {}


class BastionLabPolars:
    """Main BastionLabPolars API class.

    This class contains all the endpoints allowed on the BastionLab server for Polars.
    It is instantiated by the `bastionlab.Client` class and is accessible through the `bastionlab.Client.polars` property.
    """

    def __init__(self, client: "Client"):
        self.stub = PolarsServiceStub(client._channel)
        self.client = client

    def send_df(
        self,
        df: pl.DataFrame,
        policy: Policy = DEFAULT_POLICY,
        sanitized_columns: List[str] = [],
    ) -> "FetchableLazyFrame":
        """
        This method is used to send `pl.DataFrame` to the BastionLab server.

        It readily accepts `pl.DataFrame` and also specifies the DataFrame policy and a list of
        sensitive columns.

        Args:
            df: Polars DataFrame
            policy: BastionLab Remote DataFrame policy.
                This specifies which operations can be performed on DataFrames and they
                specified the data owner.
            sanitized_columns: This field contains (sensitive) columns in the
                DataFrame that are to be removed when a Data Scientist wishes to fetch a
                query performed on the DataFrame.

        Returns:
            FetchableLazyFrame

        """
        from .frame import FetchableLazyFrame

        self.client._refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.SendDataFrame(
                serialize_dataframe(df, policy, sanitized_columns)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(self, ref: str) -> Optional[pl.DataFrame]:
        """
        Fetches the specified `pl.DataFrame` from the BastionLab server
        with the provided reference identifier.

        Args:
            ref : str
                A unique identifier for the Remote DataFrame.

        Returns:
            Optional[pl.DataFrame]
        """

        def make_chunks_iter() -> Iterator[bytes]:
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

                yield b.data

        self.client._refresh_session_if_needed()

        try:
            df = GRPCException.map_error(
                lambda: deserialize_dataframe(make_chunks_iter())
            )
            return df
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
        """
        Executes a Composite Plan on the BastionLab server.
        A composite plan is BastionLab's internal instruction set.

        Args:
            composite_plan : str
                Serialized instructions to be executed on BastionLab server.

        Returns:
            FetchableLazyFrame
        """

        from .frame import FetchableLazyFrame

        self.client._refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.RunQuery(Query(composite_plan=composite_plan))
        )
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        """
        Enlists all the DataFrames available on the BastionLab server.

        Returns:
            List[FetchableLazyFrame]

        """
        from .frame import FetchableLazyFrame

        self.client._refresh_session_if_needed()

        res = GRPCException.map_error(lambda: self.stub.ListDataFrames(Empty()).list)
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        """
        Returns a `FetchableLazyFrame` from an BastionLab DataFrame identifier.

        Args:
            identifier : str
                A unique identifier for the Remote DataFrame.

        Returns:
            FetchableLazyFrame
        """
        from .frame import FetchableLazyFrame

        self.client._refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.GetDataFrameHeader(
                ReferenceRequest(identifier=identifier)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)

    def _persist_df(self, identifier: str):
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
        self.client._refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.PersistDataFrame(ReferenceRequest(identifier=identifier))
        )

    def _delete_df(self, identifier: str):
        """
        Deletes a Dataframe on the server from a BastionLab DataFrame identifier.

        Args
        ----
        identifier : str
            A unique identifier for the Remote DataFrame.

        Returns
        -------
        Nothing
        """
        self.client._refresh_session_if_needed()

        res = GRPCException.map_error(
            lambda: self.stub.DeleteDataFrame(ReferenceRequest(identifier=identifier))
        )

    def RemoteArray(
        self, identifier: Optional[str] = None, reference: Optional[Reference] = None
    ) -> "bastionlab.polars.frame.RemoteArray":
        if not identifier and not reference:
            raise Exception("Please pass an identifier [str, RemoteArray]")
        if reference:
            if reference:
                identifier = reference.identifier

        return RemoteArray(self, identifier)


__pdoc__["BastionLabPolars.__init__"] = False

__all__ = ["BastionLabPolars"]
