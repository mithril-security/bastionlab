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
import sys


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
    is_colab="google.colab" in sys.modules,
)


class Client:
    """
    A BastionLab client class that provides access to the BastionLab machine learning platform.
    """

    _bastionlab_torch: "BastionLabTorch" = (
        None  #: The BastionLabTorch object for accessing the torch functionality.
    )
    _bastionlab_polars: "BastionLabPolars" = (
        None  #: The BastionLabPolars object for accessing the polars functionality.
    )
    _channel: grpc.Channel  #: The underlying gRPC channel used to communicate with the server.
    __session_expiry_time: float = 0.0  #: Time in seconds
    _token: Optional[bytes] = None
    signing_key: Optional[SigningKey]

    def __init__(
        self,
        channel: grpc.Channel,
        signing_key: SigningKey,
    ):
        """
        Initializes the client with a gRPC channel to the BastionLab server.

        Args:
            channel (grpc.Channel): A gRPC channel to the BastionLab server.
        """
        self._channel = channel
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
    def torch(self) -> "BastionLabTorch":
        """
        Returns the BastionLabTorch instance used by this client.
        """
        if self._bastionlab_torch is None:
            from bastionlab.torch import BastionLabTorch

            self._bastionlab_torch = BastionLabTorch(self)
        return self._bastionlab_torch

    @property
    def polars(self) -> "BastionLabPolars":
        """
        Returns the BastionLabPolars instance used by this client.
        """
        if self._bastionlab_polars is None:
            from bastionlab.polars import BastionLabPolars

            self._bastionlab_polars = BastionLabPolars(self)
        return self._bastionlab_polars


@dataclass
class Connection:
    """
    This class represents a connection to a remote server. It holds the necessary
    information to establish and manage the connection, such as the hostname, port,
    identity (signing key), and token (if applicable).

    Attributes:
        host (str): The hostname or IP address of the remote server.
        port (int, optional): The port to use for the connection. Defaults to 50056.
        identity (SigningKey, optional): The signing key to use for authentication.
            If not provided, the connection will not be authenticated.
        channel (Any): The underlying channel object used to send and receive messages.
        token (bytes, optional): The authentication token to use for the connection.
            If not provided, the connection will not be authenticated.
        server_name (str, optional): The name of the remote server. Defaults to "bastionlab-server".
    """

    host: str
    port: Optional[int] = 50056
    identity: Optional[SigningKey] = None
    channel: Any = None
    token: Optional[bytes] = None
    _client: Optional[Client] = None  # The gRPC client object used to send messages.
    server_name: Optional[str] = "bastionlab-server"

    @staticmethod
    def _verify_user(
        server_target, server_creds, options, signing_key: Optional[SigningKey] = None
    ):

        # Set up initial connection to BastionLab for verification
        # if pubkey not known:
        #    Drop connection and fail fast authentication

        # elif known:
        #    return token and add token to channel metadata
        # """
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
        """
        Returns a `Client` instance that provides access to the BastionLab machine learning platform.
        """
        if self._client is not None:
            return self._client
        else:
            return self.__enter__()

    def close(self):
        """Closes the connection to the server.

        This method is equivalent to calling `__exit__` directly, but provides a more intuitive and readable way to close the connection.
        """
        if self._client is not None:
            self.__exit__(None, None, None)

    def __enter__(self) -> Client:
        # """Establishes a secure channel to the server and returns a `Client` object that can be used to interact with the server.
        # This method is called automatically when the `Connection` object is used in a `with` statement.
        # Returns:
        #    A `Client` object that can be used to interact with the server.
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

        auth_plugin = AuthPlugin()

        channel_cred = (
            server_creds
            if self.identity is None
            else server_creds
            if self.token is None
            else grpc.composite_channel_credentials(
                server_creds, grpc.metadata_call_credentials(auth_plugin)
            )
        )

        self.channel = grpc.secure_channel(
            server_target, channel_cred, connection_options
        )

        self._client = Client(
            self.channel,
            self.identity,
        )

        auth_plugin.client = self.client

        return self._client

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        # """Closes the connection to the server and cleans up any resources being used by the `Client` object.
        # This method is called automatically when the `with` statement is exited.
        self._client = None
        self.channel.close()
