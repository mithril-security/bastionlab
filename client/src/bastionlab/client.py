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


if TYPE_CHECKING:
    from .torch import BastionLabTorch
    from .polars import BastionLabPolars

HEART_BEAT_TICK = 25 * 60
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
    __bastionlab_torch: "BastionLabTorch" = None
    __bastionlab_polars: "BastionLabPolars" = None
    __channel: grpc.Channel
    __token: bytes

    def __init__(
        self,
        channel: grpc.Channel,
    ):
        self.__channel = channel

    @property
    def torch(self):
        if self.__bastionlab_torch is None:
            from bastionlab.torch import BastionLabTorch

            self.__bastionlab_torch = BastionLabTorch(self.__channel)
        return self.__bastionlab_torch

    @property
    def polars(self):
        if self.__bastionlab_polars is None:
            from bastionlab.polars import BastionLabPolars

            self.__bastionlab_polars = BastionLabPolars(self.__channel)
        return self.__bastionlab_polars


class AuthPlugin(grpc.AuthMetadataPlugin):
    def __init__(self, token):
        self._token = token

    def __call__(self, _, callback):
        callback((("accesstoken-bin", self._token),), None)


@dataclass
class Connection:
    host: str
    port: Optional[int] = 50056
    identity: Optional[SigningKey] = None
    channel: Any = None
    token: Optional[bytes] = None
    _client: Optional[Client] = None
    server_name: Optional[str] = "bastionlab-server"

    @staticmethod
    def _verify_user(
        server_target, server_creds, options, signing_key: Optional[SigningKey] = None
    ):
        """
        Set up initial connection to BastionLab for verification
        if pubkey not known:
            Drop connection and fail fast authentication

        elif known:
            return token and add token to channel metadata
        """
        channel = grpc.secure_channel(server_target, server_creds, options)

        session_stub = SessionServiceStub(channel)

        metadata = ()
        data: bytes = CLIENT_INFO.SerializeToString()

        if signing_key is not None:
            challenge = session_stub.GetChallenge(Empty()).value

            metadata += (("challenge-bin", challenge),)
            to_sign = b"create-session" + challenge + data

            pubkey_hex = signing_key.pubkey.hash.hex()
            signed = signing_key.sign(to_sign)
            metadata += ((f"signature-{(pubkey_hex)}-bin", signed),)

            token = session_stub.CreateSession(CLIENT_INFO, metadata=metadata).token

            return token
        else:
            session_stub.CreateSession(CLIENT_INFO)
            return None

    @property
    def client(self) -> Client:
        if self._client is not None:
            return self._client
        else:
            return self.__enter__()

    def close(self):
        if self._client is not None:
            self.__exit__(None, None, None)

    def _heart_beat(self, stub):
        while self._client is not None:
            stub.RefreshSession(Empty(), metadata=(("accesstoken-bin", self.token),))
            sleep(HEART_BEAT_TICK)

    def __enter__(self) -> Client:
        server_target = f"{self.host}:{self.port}"
        server_cert = ssl.get_server_certificate((self.host, self.port))
        server_creds = grpc.ssl_channel_credentials(
            root_certificates=bytes(server_cert, encoding="utf8")
        )
        connection_options = (("grpc.ssl_target_name_override", self.server_name),)

        # Verify user by creating session
        self.token = Connection._verify_user(
            server_target, server_creds, connection_options, self.identity
        )

        channel_cred = (
            server_creds
            if self.identity is None
            else server_creds
            if self.token is None
            else grpc.composite_channel_credentials(
                server_creds, grpc.metadata_call_credentials(AuthPlugin(self.token))
            )
        )

        self.channel = grpc.secure_channel(
            server_target, channel_cred, connection_options
        )
        stub = SessionServiceStub(self.channel)

        self._client = Client(
            self.channel,
        )

        if self.token is not None:
            daemon = Thread(
                target=self._heart_beat,
                args=(stub,),
                daemon=True,
                name="HeartBeat",
            )
            daemon.start()

        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self._client = None
        self.channel.close()
