from dataclasses import dataclass, field
import itertools
import logging
import ssl
from threading import Thread
from time import sleep
from typing import Any, List, TYPE_CHECKING, Optional
import grpc
from bastionlab.pb.bastionlab_pb2 import ReferenceRequest, Query, Empty
from bastionlab.keys import SigningKey
from grpc import StatusCode
from bastionlab.pb.bastionlab_pb2_grpc import BastionLabStub
import polars as pl
from colorama import Fore

from bastionlab.utils import (
    deserialize_dataframe,
    serialize_dataframe,
)
from bastionlab.policy import Policy, DEFAULT_POLICY
from bastionlab.errors import GRPCException


if TYPE_CHECKING:
    from bastionlab.remote_polars import RemoteLazyFrame, FetchableLazyFrame

HEART_BEAT_TICK = 25 * 60


class Client:
    def __init__(
        self,
        stub: BastionLabStub,
        token: bytes,
    ):
        self.stub = stub
        self.token = token

    def send_df(
        self, df: pl.DataFrame, policy: Policy = DEFAULT_POLICY, blacklist: List[str] = []
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(
            lambda: self.stub.SendDataFrame(serialize_dataframe(df, policy, blacklist))
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
        from bastionlab.remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(
            lambda: self.stub.RunQuery(Query(composite_plan=composite_plan))
        )
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        from bastionlab.remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(lambda: self.stub.ListDataFrames(Empty()).list)
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = GRPCException.map_error(
            lambda: self.stub.GetDataFrameHeader(
                ReferenceRequest(identifier=identifier)
            )
        )
        return FetchableLazyFrame._from_reference(self, res)


class AuthPlugin(grpc.AuthMetadataPlugin):
    def __init__(self, token):
        self._token = token

    def __call__(self, _, callback):
        callback((("accesstoken-bin", self._token),), None)


@dataclass
class Connection:
    host: str
    port: int
    signing_key: SigningKey
    channel: Any = None
    _client: Optional[Client] = None
    server_name: Optional[str] = "bastionlab-server"

    @staticmethod
    def _verify_user(server_target, server_creds, options, signing_key: SigningKey):
        """
        Set up initial connection to BastionLab for verification
        if pubkey not known:
            Drop connection and fail fast authentication

        elif known:
            return token and add token to channel metadata
        """
        channel = grpc.secure_channel(server_target, server_creds, options)

        stub = BastionLabStub(channel)

        metadata = ()

        empty_arg = Empty()
        data: bytes = empty_arg.SerializeToString()

        challenge = stub.GetChallenge(empty_arg).value

        metadata += (("challenge-bin", challenge),)
        to_sign = b"create-session" + challenge + data

        pubkey_hex = signing_key.pubkey.hash.hex()
        signed = signing_key.sign(to_sign)
        metadata += ((f"signature-{(pubkey_hex)}-bin", signed),)

        return stub.CreateSession(empty_arg, metadata=metadata).token

    @property
    def client(self) -> Client:
        if self._client is not None:
            return self._client
        else:
            return self.__enter__()

    def close(self):
        if self._client is not None:
            self.__exit__(None, None, None)

    def _heart_beat(self, stub, token):

        while True:
            stub.RefreshSession(Empty(), metadata=(("accesstoken-bin", token),))
            sleep(HEART_BEAT_TICK)

    def __enter__(self) -> Client:
        server_target = f"{self.host}:{self.port}"
        server_cert = ssl.get_server_certificate((self.host, self.port))
        server_creds = grpc.ssl_channel_credentials(
            root_certificates=bytes(server_cert, encoding="utf8")
        )
        connection_options = (("grpc.ssl_target_name_override", self.server_name),)

        # Verify user by creating session
        token = Connection._verify_user(
            server_target, server_creds, connection_options, self.signing_key
        )

        channel_cred = (
            server_creds
            if token is None
            else grpc.composite_channel_credentials(
                server_creds, grpc.metadata_call_credentials(AuthPlugin(token))
            )
        )

        self.channel = grpc.secure_channel(
            server_target, channel_cred, connection_options
        )
        stub = BastionLabStub(self.channel)

        daemon = Thread(
            target=self._heart_beat,
            args=(
                stub,
                token,
            ),
            daemon=True,
            name="HeartBeat",
        )
        daemon.start()

        self._client = Client(
            stub,
            token,
        )

        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
