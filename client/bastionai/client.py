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
    PrivacyBudget,
    NotPrivate,
    Private,
    TensorDataset,
    dataset_from_chunks,
    deserialize_weights_to_model,
    serialize_dataset,
    serialize_model,
)

from bastionai.version import __version__ as app_version

if TYPE_CHECKING:
    from bastionai.learner import RemoteLearner, RemoteDataLoader


class Client:
    """BastionAI client class."""

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
        chunk_size=100_000_000,
    ) -> Reference:
        """Uploads Pytorch Modules to BastionAI

        This endpoint transforms Pytorch modules into TorchScript modules and sends
        them to the BastionAI server.

        Args:
            model (Module): This is Pytorch's nn.Module.
            description (str): A string description of the module being uploaded.
            secret (bytes): User secret to secure module with.

        Returns:
            Reference: BastionAI reference object
        """
        return self.stub.SendModel(
            serialize_model(
                model,
                name=name,
                description=description,
                secret=secret if secret is not None else self.default_secret,
                chunk_size=chunk_size,
                client_info=self.client_info,
            )
        )

    def send_dataset(
        self,
        dataset: Dataset,
        name: str,
        description: str = "",
        secret: Optional[bytes] = None,
        privacy_limit: PrivacyBudget = NotPrivate(),
        chunk_size=100_000_000,
        batch_size=1024,
    ) -> Reference:
        """Uploads Pytorch Dataset to BastionAI.

        Args:
            model (Module): This is Pytorch's nn.Module.
            description (str): A string description of the dataset being uploaded.
            secret (bytes): User secret to secure module with.

        Returns:
            Reference: BastionAI reference object
        """
        list(
            serialize_dataset(
                dataset,
                name=name,
                description=description,
                secret=secret if secret is not None else self.default_secret,
                chunk_size=chunk_size,
                batch_size=batch_size,
                privacy_limit=privacy_limit,
                client_info=self.client_info,
            )
        )
        return self.stub.SendDataset(
            serialize_dataset(
                dataset,
                name=name,
                description=description,
                secret=secret if secret is not None else self.default_secret,
                chunk_size=chunk_size,
                batch_size=batch_size,
                privacy_limit=privacy_limit,
                client_info=self.client_info,
            )
        )

    def fetch_model_weights(self, model: Module, ref: Reference) -> None:
        """Fetchs the weights of a trained model with a BastionAI reference.

        Args:
            model (Module): This is Pytorch's nn.Module corresponding to an uploaded module.
            ref (Reference): BastionAI reference object corresponding to a module.
        """
        chunks = self.stub.FetchModule(ref)
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference) -> TensorDataset:
        """Fetchs the dataset with a BastionAI reference.

        Args:
            ref (Reference): BastionAI reference object corresponding to a dataset.

        Returns:
            TensorDataset: A wrapper to convert Tensors from BastionAI to Pytorch DataLoader
        """
        return dataset_from_chunks(self.stub.FetchDataset(ref))

    def get_available_models(self) -> List[Reference]:
        """Gets a list of references of available models on BastionAI.

        Returns:
            List[Reference]: A list of BastionAI available models as BastionAI references.
        """
        return self.stub.AvailableModels(Empty())

    def get_available_datasets(self) -> List[Reference]:
        """Gets a list of references of datasets on BastionAI.

        Returns:
            List[Reference]: A list of BastionAI available datasets as BastionAI references.
        """
        return self.stub.AvailableDatasets(Empty())

    def get_available_devices(self) -> List[str]:
        """Gets a list of devices available on BastionAI.

        Returns:
            List[str]: A list of BastionAI available devices.
        """
        return self.stub.AvailableDevices(Empty()).list

    def get_available_optimizers(self) -> List[str]:
        """Gets a list of optimizers supported by BastionAI.

        Returns:
            List[Optimizers]: A list of optimizers available on BastionAI training server.
        """
        return self.stub.AvailableOptimizers(Empty()).list

    def train(self, config: TrainConfig) -> Reference:
        """Trains a model with `TrainConfig` configuration on BastionAI.

        Args:
            config (TrainConfig): Training configuration to pass to BastionAI.
        """
        return self.stub.Train(config)

    def test(self, config: TestConfig) -> Reference:
        """Tests a dataset on a model on BastionAI.

        Args:
            config (TestConfig): Configuration for testing on BastionAI.
        """
        return self.stub.Test(config)

    def delete_dataset(self, ref: Reference) -> None:
        """Deletes a dataset with reference on BastionAI.

        Args:
            ref (Reference): BastionAI reference to dataset.
        """
        self.stub.DeleteDataset(ref)

    def delete_module(self, ref: Reference) -> None:
        """Deletes a model with reference from BastionAI.

        Args:
            ref (Reference): BastionAI reference to dataset.
        """
        self.stub.DeleteModule(ref)

    def get_metric(self, run: Reference) -> Metric:
        return self.stub.GetMetric(run)

    def RemoteDataLoader(self, *args, **kwargs) -> "RemoteDataLoader":
        """RemoteDataLoader class creates a remote dataloader on BastionAI with the training and testing datasets

        Args:
            client (Client): A BastionAI client connection
            train_dataloader (DataLoader): Dataloader serving the training dataset.
            test_dataloader (DataLoader): Dataloader serving the testing dataset.
            description (Optional[str], optional): A string description of the dataset being uploaded. Defaults to None.
            secret (Optional[bytes], optional): User secret to secure training and testing datasets with. Defaults to None.
        """
        from bastionai.learner import RemoteDataLoader

        return RemoteDataLoader(self, *args, **kwargs)

    def RemoteLearner(self, *args, **kwargs) -> "RemoteLearner":
        """A class to create a remote learner on BastionAI.

        The remote learner accepts the model to be trained and a remote dataloader created with `RemoteDataLoader`.

        Args:
            client (Client): A BastionAI client connection
            model (Union[Module, Reference]): A Pytorch nn.Module or a BastionAI model reference.
            remote_dataloader (RemoteDataLoader): A BastionAI remote dataloader.
            metric (str): Specifies the preferred loss metric.
            optimizer (OptimizerConfig): Specifies which kind of optimizer to use during training.
            device (str): Specifies on which device to train model.
            max_grad_norm (float): This specifies the clipping threshold for gradients in DP-SGD.
            model_description (Optional[str], optional): Provides additional description of models when uploading them to BastionAI server. Defaults to None.
            secret (Option[bytes], optional): User secret to secure training and testing datasets with. Defaults to None.
            expand (bool): A switch to either expand weights or not. Defaults to True.
        """
        from bastionai.learner import RemoteLearner

        return RemoteLearner(self, *args, **kwargs)


@dataclass
class Connection:
    """Connection class for creating connections to BastionAI."""

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
