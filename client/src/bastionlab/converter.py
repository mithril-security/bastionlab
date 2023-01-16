import grpc
from bastionlab.pb.bastionlab_conversion_pb2_grpc import ConversionServiceStub
from typing import Dict, Any


class BastionLabConverter:
    def __init__(self, channel: grpc.Channel) -> None:
        self._stub = ConversionServiceStub(channel)
