import getpass
from hashlib import sha256
import platform
import socket
import ssl
from bastionai.optimizer_config import *
from dataclasses import dataclass

from typing import Any, List, TYPE_CHECKING

import grpc  # type: ignore [import]

from bastionai.pb.remote_torch_pb2 import Empty, Metric, Reference, TestConfig, TrainConfig, ClientInfo  # type: ignore [import]

from bastionai.pb.remote_torch_pb2_grpc import RemoteTorchStub  # type: ignore [import]
from torch.nn import Module
from torch.utils.data import Dataset

from bastionai.utils import (
    TensorDataset,
    dataset_from_chunks,
    deserialize_weights_to_model,
    serialize_dataset,
    serialize_model,
)

from bastionai.version import __version__ as app_version
from bastionai.errors import GRPCException

if TYPE_CHECKING:
    from bastionai.learner import RemoteLearner, RemoteDataset

class Client:
    """BastionAI client class.
    
    Args:
        stub: The underlying gRPC client for the BastionAI protocol.
        client_info: Client information for telemetry purposes.
        default_secret: Default ownership secrets to be used for all requests,
                        may be overriden at the method level.
    """

    def __init__(
        self,
        stub: RemoteTorchStub,
        client_info: ClientInfo,
        default_secret: bytes = b"",
    ) -> None:
        self.stub = stub
        self.default_secret = default_secret
        self.client_info = client_info

    def send_model(
        self,
        model: Module,
        name: str,
        description: str = "",
        secret: Optional[bytes] = None,
        chunk_size: int = 100_000_000,
        progress: bool = False,
    ) -> Reference:
        """Uploads a Pytorch module to the BastionAI server.

        This endpoint transforms Pytorch modules into TorchScript modules and sends
        them to the BastionAI server over gRPC.

        Args:
            model: The Pytorch nn.Module to upload.
            name: A name for the module being uploaded.
            description: A string description of the module being uploaded.
            secret: Ownership secret override for this request.
            chunk_size: Size of a chunk in the BastionAI gRPC protocol in bytes.

        Returns:
            BastionAI gRPC protocol's reference object.
        """
        return GRPCException.map_error(lambda: self.stub.SendModel(
            serialize_model(
                model,
                name=name,
                description=description,
                secret=secret if secret is not None else self.default_secret,
                chunk_size=chunk_size,
                client_info=self.client_info,
                progress=progress,
            )
        ))

    def send_dataset(
        self,
        dataset: Dataset,
        name: str,
        description: str = "",
        secret: Optional[bytes] = None,
        privacy_limit: Optional[float] = None,
        chunk_size: int = 100_000_000,
        batch_size: int = 1024,
        train_dataset: Optional[Reference] = None,
        progress: bool = False,
    ) -> Reference:
        """Uploads a Pytorch Dataset to the BastionAI server.

        Args:
            model: The Pytorch Dataset to upload.
            name: A name for the dataset being uploaded.
            description: A string description of the dataset being uploaded.
            secret: Ownership secret override for this request.
            chunk_size: Size of a chunk in the BastionAI gRPC protocol in bytes.
            batch_size: Size of a unit of serialization in number of samples,
                        increasing this value may increase serialization throughput
                        at the price of a higher memory consumption.
            train_dataset: metadata, True means this dataset is suited for training,
                   False that it should be used for testing/validating only

        Returns:
            BastionAI gRPC protocol's reference object.
        """
        return GRPCException.map_error(lambda: self.stub.SendDataset(
            serialize_dataset(
                dataset,
                name=name,
                description=description,
                secret=secret if secret is not None else self.default_secret,
                chunk_size=chunk_size,
                batch_size=batch_size,
                privacy_limit=privacy_limit,
                client_info=self.client_info,
                train_dataset=train_dataset,
                progress=progress,
            )
        ))

    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        """Fetches the weights of a distant trained model with a BastionAI gRPC protocol reference
        and loads the weights into the passed model instance.

        Args:
            model: The Pytorch's nn.Module whose weights will be replaced by the fetched weights.
            ref: BastionAI gRPC protocol reference object corresponding to the distant trained model.
        """
        chunks = GRPCException.map_error(lambda: self.stub.FetchModule(ref))
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference) -> TensorDataset:
        """Fetches the distant dataset with a BastionAI gRPC protocol reference.

        Args:
            ref: BastionAI gRPC protocol reference object corresponding to the distant dataset.

        Returns:
            A dataset instance built from received data.
        """
        return dataset_from_chunks(GRPCException.map_error(lambda: self.stub.FetchDataset(ref)))

    def get_available_models(self) -> List[Reference]:
        """Returns the list of BastionAI gRPC protocol references of all available models on the server."""
        return GRPCException.map_error(lambda: self.stub.AvailableModels(Empty())).list

    def get_available_datasets(self) -> List[Reference]:
        """Returns the list of BastionAI gRPC protocol references of all datasets on the server."""
        return GRPCException.map_error(lambda: self.stub.AvailableDatasets(Empty())).list

    def get_available_devices(self) -> List[str]:
        """Returns the list of devices available on the server."""
        return GRPCException.map_error(lambda: self.stub.AvailableDevices(Empty())).list

    def get_available_optimizers(self) -> List[str]:
        """Returns the list of optimizers supported by the server."""
        return GRPCException.map_error(lambda: self.stub.AvailableOptimizers(Empty())).list

    def train(self, config: TrainConfig) -> Reference:
        """Trains a model with hyperparameters defined in `config` on the BastionAI server.

        Args:
            config: Training configuration that specifies the model, dataset and hyperparameters.
        """
        return GRPCException.map_error(lambda: self.stub.Train(config))

    def test(self, config: TestConfig) -> Reference:
        """Tests a dataset on a model according to `config` on the BastionAI server.

        Args:
            config: Testing configuration that specifies the model, dataset and hyperparameters.
        """
        return GRPCException.map_error(lambda: self.stub.Test(config))

    def delete_dataset(self, ref: Reference) -> None:
        """Deletes the dataset correponding to the given `ref` reference on the BastionAI server.

        Args:
            ref: BastionAI gRPC protocol reference of the dataset to be deleted.
        """
        GRPCException.map_error(lambda: self.stub.DeleteDataset(ref))

    def delete_module(self, ref: Reference) -> None:
        """Deletes the module correponding to the given `ref` reference on the BastionAI server.

        Args:
            ref: BastionAI gRPC protocol reference of the module to be deleted.
        """
        GRPCException.map_error(lambda: self.stub.DeleteModule(ref))

    def get_metric(self, run: Reference) -> Metric:
        """Returns the value of the metric associated with the given `run` reference.

        Args:
            run: BastionAI gRPC protocol reference of the run whose metric is read.
        """
        return GRPCException.map_error(lambda: self.stub.GetMetric(run))
    
    def list_remote_datasets(self) -> List["RemoteDataset"]:
        from bastionai.learner import RemoteDataset

        return RemoteDataset.list_available(self)

    def RemoteDataset(self, *args, **kwargs) -> "RemoteDataset":
        """Returns a RemoteDataLoader object encapsulating a training and testing dataloaders
        on the remote server that uses this client to communicate with the server.

        Args:
            *args: all arguments are forwarded to the `RemoteDataLoader` constructor.
            **kwargs: all keyword arguments are forwarded to the `RemoteDataLoader` constructor.
        """
        from bastionai.learner import RemoteDataset

        return RemoteDataset(self, *args, **kwargs)

    def RemoteLearner(self, *args, **kwargs) -> "RemoteLearner":
        """Returns a RemoteLearner object encapsulating a model and hyperparameters for
        training and testing on the remote server and that uses this client to communicate with the server.

        Args:
            *args: all arguments are forwarded to the `RemoteDataLoader` constructor.
            **kwargs: all keyword arguments are forwarded to the `RemoteDataLoader` constructor.
        """
        from bastionai.learner import RemoteLearner

        return RemoteLearner(self, *args, **kwargs)


@dataclass
class Connection:
    """Context manger that handles a connection to a BastionAI server.
    It returns a `Client` to use the connexion within its context.
    
    Args:
        host: Hostname of the BastionAI server.
        port: Port of the BastionAI server.
        default_secret: Default owner secret passed to the constructor of the return `Client`.
    """

    host: str
    port: int
    default_secret: bytes = b""  # we don't use the secrets yet
    channel: Any = None
    server_name: str = "bastionai-srv"

    def __enter__(self) -> Client:
        uname = platform.uname()
        client_info = ClientInfo(
            uid=sha256((socket.gethostname() + "-" + getpass.getuser()).encode("utf-8"))
            .digest()
            .hex(),
            platform_name=uname.system,
            platform_arch=uname.machine,
            platform_version=uname.version,
            platform_release=uname.release,
            user_agent="bastionai_python",
            user_agent_version=app_version,
        )
        connection_options = (("grpc.ssl_target_name_override", self.server_name),)
        server_cert = ssl.get_server_certificate((self.host, self.port))

        server_cred = grpc.ssl_channel_credentials(
            root_certificates=bytes(server_cert, encoding="utf8")
        )

        server_target = f"{self.host}:{self.port}"
        self.channel = grpc.secure_channel(
            server_target, server_cred, options=connection_options
        )
        return Client(RemoteTorchStub(self.channel), client_info=client_info, default_secret=self.default_secret)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.channel.close()
