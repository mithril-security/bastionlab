from dataclasses import dataclass
import socket
import ssl
from typing import Any, List, Optional
import os
import grpc
from grpc import Channel, RpcError, secure_channel
from bastionai.utils.errors import check_rpc_exception, check_socket_exception
from pb.remote_torch_pb2 import (Empty, Reference, TestConfig,
                                 TrainConfig)
from pb.remote_torch_pb2_grpc import RemoteTorchStub
from torch.nn import Module
from torch.utils.data import Dataset

from bastionai.utils.utils import (ArtifactDataset, deserialize_weights_to_model, metric_tqdm, metric_tqdm_with_epochs,
                                   serialize_dataset, serialize_model)
from functools import wraps

CONNECTION_TIMEOUT = 10


def raise_exception_if_conn_closed(f):
    """
    Decorator which raises an exception if the BastionAiConnection is closed before calling
    the decorated method
    """

    @wraps(f)
    def wrapper(self, *args, **kwds):
        if self.closed:
            raise ValueError("Illegal operation on closed connection.")
        return f(self, *args, **kwds)

    return wrapper


class Connection:
    def __init__(self, addr: str, port: int, server_name: str) -> None:
        client_to_server = f"{addr}:{port}"
        try:
            socket.setdefaulttimeout(CONNECTION_TIMEOUT)
            server_cert = ssl.get_server_certificate(
                (addr, port)
            )
            server_credentials = grpc.ssl_channel_credentials(
                root_certificates=bytes(
                    server_cert, encoding="utf8")
            )

        except grpc.RpcError as rpc_error:
            raise ConnectionError(check_rpc_exception(rpc_error))

        except socket.error as socket_error:
            raise ConnectionError(check_socket_exception(socket_error))

        connection_options = (("grpc.ssl_target_name_override", server_name),)

        try:
            channel = secure_channel(
                client_to_server, server_credentials, options=connection_options)
            self._stub = RemoteTorchStub(channel)
            self._channel = channel
        except RpcError as rpc_error:
            channel.close()
            raise ConnectionError(check_rpc_exception(rpc_error))


@dataclass
class BastionAiClient:
    """BastionAI client class."""

    _stub: RemoteTorchStub
    _channel: Optional[Channel] = None

    closed: bool = False

    def __init__(self, addr: str, port: int = 50053, server_name: str = 'bastionai-srv', debug_mode: bool = False) -> None:
        """Connect to the server with the specified parameters.
        Currently, the server is set up in Simulation mode and you do not need to provide your own server certificate.

        Args:
            addr (str): The address of BastionAI server you want to reach.
            server_name (str, optional): Contains the CN expected by the server TLS certificate. Defaults to "bastionai-srv".
            port (int, optional):  Connection server port. Defaults to 50053.

        Raises:
            ConnectionError: will be raised if the connection with the server fails.
        """
        if debug_mode:  # pragma: no cover
            os.environ["GRPC_TRACE"] = "transport_security,tsi"
            os.environ["GRPC_VERBOSITY"] = "DEBUG"

        Connection(addr, port, server_name)

    @raise_exception_if_conn_closed
    def send_model(self, model: Module, description: str, secret: bytes) -> Reference:
        """Uploads Pytorch Modules to BastionAI

        This endpoint transforms Pytorch modules into TorchScript and sends
        them to the BastionAI server.

        Args:
            model (torch.nn.modules.module.Module): This is Pytorch's nn.Module.
            description (str): A string description of the module being uploaded.
            secret (bytes): User secret to secure module with.

        Returns:
            Reference: BastionAI reference object
        """
        return self._stub.SendModel(serialize_model(model, description=description, secret=secret))

    @raise_exception_if_conn_closed
    def send_dataset(self, dataset: Dataset, description: str, secret: bytes) -> Reference:
        """Uploads Pytorch Dataset to BastionAI.

        This endpoint transforms Pytorch Datasets into TorchScript and sends
        them to the BastionAI server.

        Args:
            dataset (torch.utils.data.dataset.Dataset): This is Pytorch's Dataset.
            description (str): A string description of the dataset being uploaded.
            secret (bytes): User secret to secure module with.

        Returns:
            Reference: BastionAI reference object
        """
        return self._stub.SendDataset(serialize_dataset(dataset, description=description, secret=secret))

    @raise_exception_if_conn_closed
    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        """Fetchs the weights of a trained model with a BastionAI reference.

        Args:
            model (torch.nn.modules.module.Module): This is Pytorch's nn.Module corresponding to an uploaded module.
            ref (remote_torch_pb2.Reference): BastionAI reference object corresponding to a module.
        """
        chunks = self._stub.FetchModule(ref)
        deserialize_weights_to_model(model, chunks)

    @raise_exception_if_conn_closed
    def fetch_dataset(self, ref: Reference) -> ArtifactDataset:
        """Fetchs the dataset with a BastionAI reference.

        Args:
            ref (remote_torch_pb2.Reference): BastionAI reference object corresponding to a dataset.

        Returns:
            ArtifactDataset: A wrapper to convert Tensors from BastionAI to Pytorch DataLoader
        """
        return ArtifactDataset(self._stub.FetchDataset(ref))

    @raise_exception_if_conn_closed
    def get_available_models(self) -> List[Reference]:
        """Gets a list of references of available models on BastionAI.

        Returns: 
            A list of BastionAI available model references.
        """
        return self._stub.AvailableModels(Empty())

    @raise_exception_if_conn_closed
    def get_available_datasets(self) -> List[Reference]:
        """Gets a list of references of datasets on BastionAI.


        Returns: 
            A list of BastionAI available dataset references.
        """
        return self._stub.AvailableDatasets(Empty())

    @raise_exception_if_conn_closed
    def get_available_devices(self) -> List[str]:
        """Gets a list of devices available on BastionAI.

        Returns: 
            A list of BastionAI available devices.
        """
        return self._stub.AvailableDevices(Empty()).list

    @raise_exception_if_conn_closed
    def get_available_optimizers(self) -> List[str]:
        """Gets a list of optimizers supported by BastionAI.

        Returns:
            A list of optimizers available on BastionAI training server.
        """
        return self._stub.AvailableOptimizers(Empty()).list

    @raise_exception_if_conn_closed
    def train(self, config: TrainConfig) -> None:
        """Trains a model with `TrainConfig` configuration on BastionAI.

        Args:
            config (remote_torch_pb2.TrainConfig): Training configuration to pass to BastionAI.
        """
        metric_tqdm_with_epochs(self._stub.Train(
            config), name=f"loss ({config.metric})")

    @raise_exception_if_conn_closed
    def test(self, config: TestConfig) -> float:
        """Tests a dataset on a model on BastionAI.

        Args:
            config (remote_torch_pb2.TestConfig): Configuration for testing on BastionAI.

        Returns:
            float: The evaluation of the model on the datatset.
        """
        metric_tqdm(self._stub.Test(config), name=f"metric ({config.metric})")

    @raise_exception_if_conn_closed
    def delete_dataset(self, ref: Reference) -> None:
        """Deletes a dataset with reference on BastionAI.

        Args:
            ref (remote_torch_pb2.Reference): BastionAI reference to dataset.
        """
        self._stub.DeleteDataset(ref)

    @raise_exception_if_conn_closed
    def delete_module(self, ref: Reference) -> None:
        """Deletes a model with reference from BastionAI.

        Args:
            ref (remote_torch_pb2.Reference): BastionAI reference to dataset.
        """
        self._stub.DeleteModule(ref)

    def close(self):
        """Close the connection between the client and the training server. This method has no effect if the connection is already closed."""
        if not self.closed:
            self._channel.close()
            self.closed = True
            self._channel = None
            self._stub = None


@wraps(BastionAiClient.__init__, assigned=("__doc__", "__annotations__"))
def connect(*args, **kwargs):
    return BastionAiClient(*args, **kwargs)
