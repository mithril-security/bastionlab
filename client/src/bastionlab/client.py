from dataclasses import dataclass, field
import itertools
import logging
from threading import Thread
from time import sleep
from typing import Any, List, TYPE_CHECKING, Optional
import grpc
from bastionlab.pb.bastionlab_pb2 import ReferenceRequest, Query, Empty
from bastionlab.keys import SigningKey
from bastionlab.pb.bastionlab_pb2_grpc import BastionLabStub
from bastionlab.errors import GRPCException
import polars as pl

from bastionlab.utils import (
    deserialize_dataframe,
    serialize_dataframe,
)

if TYPE_CHECKING:
    from bastionlab.remote_polars import RemoteLazyFrame, FetchableLazyFrame

HEART_BEAT_TICK = 25 * 60

class Client:
    def __make_grpc_call(
        self,
        call: str,
        arg: Any,
    ) -> Any:
        metadata = (("accesstoken-bin", self.token),)

        # todo challenges
        logging.debug(f"GRPC Call {call}; using metadata {metadata}")

        fn = getattr(self.stub, call)
        return GRPCException.map_error(lambda: fn(arg, metadata=metadata))

    def __init__(
        self,
        stub: BastionLabStub,
        token: bytes,
        default_signing_keys: List[SigningKey] = [],
    ):
        self.stub = stub
        self.token = token
        self.default_signing_keys = default_signing_keys

    def send_df(self, df: pl.DataFrame) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.SendDataFrame(serialize_dataframe(df))
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(
        self,
        ref: List[str],
    ) -> pl.DataFrame:

        joined_bytes = b""
        for b in self.__make_grpc_call(
            "FetchDataFrame",
            ReferenceRequest(identifier=ref),
        ):
            joined_bytes += b.data

        return deserialize_dataframe(joined_bytes)

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.__make_grpc_call(
            "RunQuery",
            Query(composite_plan=composite_plan),
        )
        return FetchableLazyFrame._from_reference(self, res)

    def list_dfs(self) -> List["FetchableLazyFrame"]:
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.ListDataFrames(Empty()).list
        return [FetchableLazyFrame._from_reference(self, ref) for ref in res]

    def get_df(self, identifier: str) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.__make_grpc_call(
            "GetDataFrameHeader",
            ReferenceRequest(identifier=identifier),
        )
        return FetchableLazyFrame._from_reference(self, res)


def verify_user(server_target, signing_keys: List[SigningKey] = []):
    # Set up initial connection to BastionLab for verification
    # Drop if pubkey not known, if know, return token and add token to channel metadata
    channel = grpc.insecure_channel(server_target)

    stub = BastionLabStub(channel)

    metadata = ()

    empty_arg = Empty()
    data: bytes = empty_arg.SerializeToString()

    challenge = GRPCException.map_error(lambda: stub.GetChallenge(empty_arg)).value

    metadata += (("challenge-bin", challenge),)
    to_sign = b"create-session" + challenge + data

    for k in signing_keys:
        pubkey_hex = k.pubkey.hash.hex()
        signed = k.sign(to_sign)
        metadata += ((f"signature-{(pubkey_hex)}-bin", signed),)

    return stub.CreateSession(empty_arg, metadata=metadata).token


class AuthPlugin(grpc.AuthMetadataPlugin):
    def __init__(self, token):
        self._token = token

    def __call__(self, _, callback):
        callback((("accesstoken-bin", self._token),), None)


@dataclass
class Connection:
    host: str
    port: int
    license_key: Optional[SigningKey]
    channel: Any = None
    _client: Optional[Client] = None
    default_signing_keys: List[SigningKey] = field(default_factory=list)

    @property
    def client(self) -> Client:
        if self._client is not None:
            return self._client
        else:
            return self.__enter__()

    def close(self):
        if self._client is not None:
            self.__exit__(None, None, None)

    def heart_beat(self, stub, token):

        while True:
            stub.RefreshSession(Empty(), metadata=(("accesstoken-bin", token),))
            sleep(HEART_BEAT_TICK)

    def __enter__(self) -> Client:
        server_target = f"{self.host}:{self.port}"

        if self.license_key is not None:
            if self.license_key not in self.default_signing_keys:
                self.default_signing_keys.append(self.license_key)

            # Verify user by creating session
        token = verify_user(server_target, self.default_signing_keys)
        # connection_options = (("grpc.ssl_target_name_override", self.server_name),)
        # server_cert = ssl.get_server_certificate((self.host, self.port))

        # server_cred = ssl_channel_credentials(
        #     root_certificates=bytes(server_cert, encoding="utf8")
        # )

        # channel_cred = (
        #     server_cred
        #     if self._jwt is None
        #     else composite_channel_credentials(
        #         server_cred, metadata_call_credentials(AuthPlugin(self._jwt))
        #     )
        # )

        self.channel = grpc.insecure_channel(server_target)
        stub = BastionLabStub(self.channel)

        daemon = Thread(target=self.heart_beat, args=(stub, token,), daemon=True, name='HeartBeat')
        daemon.start()

        self._client = Client(
            stub,
            token,
            default_signing_keys=self.default_signing_keys,
        )

        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
