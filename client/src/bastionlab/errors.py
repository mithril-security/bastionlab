import grpc  # type: ignore [import]
from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous  # type: ignore [import]
from dataclasses import dataclass
from typing import Callable, TypeVar, Union


T = TypeVar("T")


@dataclass
class GRPCException(Exception):
    """
    A custom exception class for wrapping gRPC errors. This allows for better error
    handling and display.

    Attributes:
        err (Union[grpc._channel._InactiveRpcError, grpc._channel._MultiThreadedRendezvous]):
            The gRPC error that was caught and wrapped by this exception.
    """

    err: Union[grpc._channel._InactiveRpcError, grpc._channel._MultiThreadedRendezvous]

    @property
    def code(self) -> grpc.StatusCode:
        """
        Get the status code of the gRPC error.

        Returns:
            The status code of the gRPC error.
        """
        return self.err._state.code

    def __str__(self):
        if self.code == grpc.StatusCode.NOT_FOUND:
            prefix = "Remote resource not found"
        elif self.code == grpc.StatusCode.INVALID_ARGUMENT:
            prefix = "Invalid argument passed to server"
        elif self.code == grpc.StatusCode.OUT_OF_RANGE:
            prefix = "Not yet available (or out-of-range) resource"
        elif self.code == grpc.StatusCode.INTERNAL:
            prefix = "Internal server error"
        elif self.code == grpc.StatusCode.CANCELLED:
            prefix = f"Cancelled gRPC call"
        elif self.code == grpc.StatusCode.UNAVAILABLE:
            prefix = "Connection to the gRPC server failed"
        elif self.code == grpc.StatusCode.UNIMPLEMENTED:
            prefix = "Incompatible client/server versions"
        elif self.code == grpc.StatusCode.FAILED_PRECONDITION:
            prefix = "Attestation is not available. Running in Simulation Mode"
        else:
            prefix = f"Received gRPC error"

        return f"{prefix}: code={self.code} message={self.err.details()}"

    @staticmethod
    def map_error(f: Callable[[], T]) -> T:
        """
        Map gRPC errors to `GRPCException` exceptions.

        Args:
            f: The function to call and map errors from.

        Returns:
            The result of calling `f`, if no errors were raised.

        Raises:
            GRPCException: if `f` raised a gRPC error.
        """
        try:
            return f()
        except _InactiveRpcError as e:
            raise GRPCException(e)
        except _MultiThreadedRendezvous as e:
            raise GRPCException(e)
