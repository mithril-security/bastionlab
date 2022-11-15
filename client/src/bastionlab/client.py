from dataclasses import dataclass, field
import itertools
import logging
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


class Client:
    def __make_grpc_call(
        self,
        call: str,
        arg: Any,
        streaming: bool = False,
        method: Optional[bytes] = None,
        signing_keys: List[SigningKey] = [],
    ) -> Any:
        # does not support streaming calls for now

        needs_signing_and_challenge = method is not None and (
            len(signing_keys) > 0 or len(self.default_signing_keys) > 0
        )

        metadata = ()
        if needs_signing_and_challenge:
            data: bytes = arg.SerializeToString()

            logging.debug(f"GRPC Call {call}: getting a challenge")
            challenge = GRPCException.map_error(
                lambda: self.stub.GetChallenge(Empty())
            ).value

            metadata += (("challenge-bin", challenge),)
            to_sign = method + challenge + data

            keys = ()
            for k in itertools.chain(signing_keys, self.default_signing_keys):
                signed = k.sign(to_sign)
                pubkey_hex = k.pubkey.hash.hex()
                metadata += ((f"signature-{(pubkey_hex)}-bin", signed),)
                keys += pubkey_hex

            metadata += (("signing-keys", keys),)

        # todo challenges
        logging.debug(f"GRPC Call {call}; using metadata {metadata}")

        fn = getattr(self.stub, call)
        return GRPCException.map_error(lambda: fn(arg, metadata=metadata))

    def __init__(
        self, stub: BastionLabStub, default_signing_keys: List[SigningKey] = []
    ):
        self.stub = stub
        self.default_signing_keys = default_signing_keys

    def send_df(self, df: pl.DataFrame) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.SendDataFrame(serialize_dataframe(df))
        return FetchableLazyFrame._from_reference(self, res)

    def _fetch_df(
        self, ref: List[str], signing_keys: List[SigningKey] = []
    ) -> pl.DataFrame:
        joined_bytes = b""
        for b in self.__make_grpc_call(
            "FetchDataFrame",
            ReferenceRequest(identifier=ref),
            method=b"fetch",
            signing_keys=signing_keys,
        ):
            joined_bytes += b.data

        return deserialize_dataframe(joined_bytes)

    def _run_query(
        self,
        composite_plan: str,
    ) -> "FetchableLazyFrame":
        from bastionlab.remote_polars import FetchableLazyFrame

        res = self.stub.RunQuery(Query(composite_plan=composite_plan))
        return FetchableLazyFrame._from_reference(self, res)


@dataclass
class Connection:
    host: str
    port: int
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

    def __enter__(self) -> Client:
        server_target = f"{self.host}:{self.port}"
        self.channel = grpc.insecure_channel(server_target)
        self._client = Client(
            BastionLabStub(self.channel),
            default_signing_keys=self.default_signing_keys,
        )
        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
