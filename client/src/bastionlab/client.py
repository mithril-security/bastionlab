from dataclasses import dataclass
import ssl
from threading import Thread
from time import sleep
from typing import Any, TYPE_CHECKING, Optional
from hashlib import sha256
import grpc
from .keys import SigningKey
from .pb.bastionlab_pb2 import Empty
from .pb.bastionlab_pb2 import ClientInfo
from .version import __version__ as app_version
from .pb.bastionlab_pb2_grpc import SessionServiceStub
import platform
import socket
import getpass
import time
import logging


if TYPE_CHECKING:
    from .torch import BastionLabTorch
    from .polars import BastionLabPolars


class AuthPlugin(grpc.AuthMetadataPlugin):
    client: Optional["Client"] = None

    def __call__(self, _, callback):
        if self.client is not None and self.client._token is not None:
            callback((("accesstoken-bin", self.client._token),), None)
        else:
            callback((), None)


UNAME = platform.uname()
CLIENT_INFO = ClientInfo(
    uid=sha256((socket.gethostname() + "-" + getpass.getuser()).encode("utf-8"))
    .digest()
    .hex(),
    platform_name=UNAME.system,
    platform_arch=UNAME.machine,
    platform_version=UNAME.version,
    platform_release=UNAME.release,
    user_agent="bastionlab_python",
    user_agent_version=app_version,
)


class Client:
    channel: grpc.Channel

    __session_stub: SessionServiceStub
    __session_expiry_time: float = 0.0  # time in seconds
    _token: Optional[bytes] = None

    __bastionlab_torch: Optional["BastionLabTorch"] = None
    __bastionlab_polars: Optional["BastionLabPolars"] = None

    signing_key: Optional[SigningKey]

    def __init__(
        self,
        channel: grpc.Channel,
        signing_key: SigningKey,
    ):
        self.channel = channel
        self.__session_stub = SessionServiceStub(channel)
        self.signing_key = signing_key

    def refresh_session_if_needed(self):
        current_time = time.time()

        if current_time > self.__session_expiry_time:
            self._token = None
            self.__create_session()

    def __create_session(self):
        logging.debug("Refreshing session.")

        metadata = ()
        if self.signing_key is not None:
            data: bytes = CLIENT_INFO.SerializeToString()
            challenge = self.__session_stub.GetChallenge(Empty()).value

            metadata += (("challenge-bin", challenge),)
            to_sign = b"create-session" + challenge + data

            pubkey_hex = self.signing_key.pubkey.hash.hex()
            signed = self.signing_key.sign(to_sign)
            metadata += ((f"signature-{pubkey_hex}-bin", signed),)

        res = self.__session_stub.CreateSession(CLIENT_INFO, metadata=metadata)

        # So, just to be sure, we refresh our token early (30s).
        adjusted_expiry_delay = max(res.expiry_time - 30_000, 0)

        self.__session_expiry_time = (
            time.time() + adjusted_expiry_delay / 1000  # convert to seconds
        )
        self._token = res.token

    @property
    def torch(self):
        if self.__bastionlab_torch is None:
            from bastionlab.torch import BastionLabTorch

            self.__bastionlab_torch = BastionLabTorch(self)
        return self.__bastionlab_torch

    @property
    def polars(self):
        if self.__bastionlab_polars is None:
            from bastionlab.polars import BastionLabPolars

            self.__bastionlab_polars = BastionLabPolars(self)
        return self.__bastionlab_polars


@dataclass
class Connection:
    host: str
    port: Optional[int] = 50056
    identity: Optional[SigningKey] = None
    channel: Any = None
    token: Optional[bytes] = None
    _client: Optional[Client] = None
    server_name: Optional[str] = "bastionlab-server"

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
        server_cert = ssl.get_server_certificate((self.host, self.port))
        server_creds = grpc.ssl_channel_credentials(
            root_certificates=bytes(server_cert, encoding="utf8")
        )
        connection_options = (("grpc.ssl_target_name_override", self.server_name),)

        auth_plugin = AuthPlugin()
        channel_cred = grpc.composite_channel_credentials(
            server_creds, grpc.metadata_call_credentials(auth_plugin)
        )

        self.channel = grpc.secure_channel(
            server_target, channel_cred, connection_options
        )

        self._client = Client(
            self.channel,
            signing_key=self.identity,
        )

        auth_plugin.client = self.client

        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
