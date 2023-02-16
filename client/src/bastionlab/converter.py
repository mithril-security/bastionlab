import grpc
from bastionlab.pb.bastionlab_conversion_pb2_grpc import ConversionServiceStub
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import Client


class BastionLabConverter:
    def __init__(self, client: "Client") -> None:
        self._stub = ConversionServiceStub(client._channel)
