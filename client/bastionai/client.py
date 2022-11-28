import getpass
from hashlib import sha256
import platform
import socket
import ssl
from bastionai.optimizer_config import *
from dataclasses import dataclass
import logging
import itertools

from typing import Any, List, Optional, TYPE_CHECKING, Union

import grpc  # type: ignore [import]

from bastionai.pb.remote_torch_pb2 import Empty, MetricResponse, TestRequest, TrainRequest, ClientInfo  # type: ignore [import]
from bastionai.pb.remote_torch_pb2_grpc import RemoteTorchStub  # type: ignore [import]
from bastionai.license import LicenseBuilder, License
from torch.nn import Module
from torch.utils.data import Dataset

from bastionai.utils import (
    Reference,
    TensorDataset,
    dataset_from_chunks,
    module_from_chunks,
    deserialize_weights_to_model,
    serialize_dataset,
    serialize_model,
)

from bastionai.keys import SigningKey

from bastionai.version import __version__ as app_version
from bastionai.errors import GRPCException

if TYPE_CHECKING:
    from bastionai.learner import RemoteLearner, RemoteDataset

class Client:
    """BastionAI client class.
    
    Args:
        stub: The underlying gRPC client for the BastionAI protocol.
        client_info: Client information for telemetry purposes.
        default_license: Default license to be used with all the objects
                        sent over this connection,
                        may be overriden at the method level.
    """

    def __make_grpc_call(self, call: str, arg: Any, streaming: bool = False, method: Optional[bytes] = None, signing_keys: List[SigningKey] = []) -> Any:
        # does not support streaming calls for now

        needs_signing_and_challenge = method is not None and (len(signing_keys) > 0 or len(self.default_signing_keys) > 0)

        metadata = ()
        if needs_signing_and_challenge:
            data: bytes = arg.SerializeToString()

            logging.debug(f"GRPC Call {call}: getting a challenge")
            challenge = GRPCException.map_error(lambda: self.stub.GetChallenge(Empty())).value

            metadata += (("challenge-bin", challenge),)
            to_sign = method + challenge + data

            for k in itertools.chain(signing_keys, self.default_signing_keys):
                signed = k.sign(to_sign)
                metadata += ((f"signature-{(k.pubkey.hash.hex())}-bin", signed),)

        # todo challenges
        logging.debug(f"GRPC Call {call}; using metadata {metadata}")

        fn = getattr(self.stub, call)
        return GRPCException.map_error(lambda: fn(arg, metadata=metadata))

    def __init__(
        self,
        stub: RemoteTorchStub,
        client_info: ClientInfo,
        default_license: Optional[Union[LicenseBuilder, License]] = None,
        default_signing_keys: List[SigningKey] = []
    ) -> None:
        self.stub = stub
        self.default_license = default_license
        self.client_info = client_info
        self.default_signing_keys = default_signing_keys

    def send_model(
        self,
        model: Module,
        name: str,
        description: str = "",
        license: Optional[Union[LicenseBuilder, License]] = None,
        chunk_size: int = 4_194_285,
        progress: bool = False,
    ) -> Reference:
        """Uploads a Pytorch module to the BastionAI server.

        This endpoint transforms Pytorch modules into TorchScript modules and sends
        them to the BastionAI server over gRPC.

        Args:
            model: The Pytorch nn.Module to upload.
            name: A name for the module being uploaded.
            description: A string description of the module being uploaded.
            license: Ownership license override for this request.
            chunk_size: Size of a chunk in the BastionAI gRPC protocol in bytes.

        Returns:
            BastionAI gRPC protocol's reference object.
        """
        li = license or self.default_license
        if li is None:
            raise ValueError("You must specify an access-control license")

        return Reference(GRPCException.map_error(lambda: self.stub.SendModel(
            serialize_model(
                model,
                name=name,
                description=description,
                license=(li.build() if isinstance(li, LicenseBuilder) else li).ser(),
                chunk_size=chunk_size,
                client_info=self.client_info,
                progress=progress,
            )
        )))

    def send_dataset(
        self,
        dataset: Dataset,
        name: str,
        description: str = "",
        license: Optional[Union[LicenseBuilder, License]] = None,
        privacy_limit: Optional[float] = None,
        chunk_size: int = 4_194_285,
        batch_size: int = 1024,
        train_dataset: Optional[Reference] = None,
        progress: bool = False,
    ) -> Reference:
        """Uploads a Pytorch Dataset to the BastionAI server.

        Args:
            model: The Pytorch Dataset to upload.
            name: A name for the dataset being uploaded.
            description: A string description of the dataset being uploaded.
            license: Ownership license override for this request.
            chunk_size: Size of a chunk in the BastionAI gRPC protocol in bytes.
            batch_size: Size of a unit of serialization in number of samples,
                        increasing this value may increase serialization throughput
                        at the price of a higher memory consumption.
            train_dataset: metadata, True means this dataset is suited for training,
                   False that it should be used for testing/validating only

        Returns:
            BastionAI gRPC protocol's reference object.
        """
        li = license or self.default_license
        if li is None:
            raise ValueError("You must specify an access-control license")

        return Reference(GRPCException.map_error(lambda: self.stub.SendDataset(
            serialize_dataset(
                dataset,
                name=name,
                description=description,
                license=(li.build() if isinstance(li, LicenseBuilder) else li).ser(),
                chunk_size=chunk_size,
                batch_size=batch_size,
                privacy_limit=privacy_limit,
                client_info=self.client_info,
                train_dataset=train_dataset,
                progress=progress,
            )
        )))

    def load_checkpoint(self, model: Module, ref: Reference, signing_keys: List[SigningKey] = []) -> None:
        """Fetches the weights of a distant trained model with a BastionAI gRPC protocol reference
        and loads the weights into the passed model instance.

        Args:
            model: The Pytorch's nn.Module whose weights will be replaced by the fetched weights.
            ref: BastionAI gRPC protocol reference object corresponding to the distant trained model.
        """
        chunks = self.__make_grpc_call("FetchCheckpoint", ref.request(), method=b"fetch", signing_keys=signing_keys)
        deserialize_weights_to_model(model, chunks)

    def fetch_dataset(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> TensorDataset:
        """Fetches the distant dataset with a BastionAI gRPC protocol reference.

        Args:
            ref: BastionAI gRPC protocol reference object corresponding to the distant dataset.

        Returns:
            A dataset instance built from received data.
        """
        return dataset_from_chunks(self.__make_grpc_call("FetchDataset", ref.request(), method=b"fetch", signing_keys=signing_keys))

    def fetch_model(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> Module:
        return module_from_chunks(
            self.__make_grpc_call("FetchModel", ref.request(), method=b"fetch", signing_keys=signing_keys))
    
    def fetch_run(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> Module:
        return self.__make_grpc_call("FetchRun", ref.request(), method=b"fetch", signing_keys=signing_keys)

    def get_available_models(self, *, signing_keys: List[SigningKey] = []) -> List[Reference]:
        """Returns the list of BastionAI gRPC protocol references of all available models on the server."""
        return [Reference(x) for x in self.__make_grpc_call("AvailableModels", Empty(), method=b"list", signing_keys=signing_keys).list]

    def get_available_datasets(self, *, signing_keys: List[SigningKey] = []) -> List[Reference]:
        """Returns the list of BastionAI gRPC protocol references of all datasets on the server."""
        return [Reference(x) for x in self.__make_grpc_call("AvailableDatasets", Empty(), method=b"list", signing_keys=signing_keys).list]
    
    def get_available_checkpoints(self, *, signing_keys: List[SigningKey] = []) -> List[Reference]:
        """Returns the list of BastionAI gRPC protocol references of all datasets on the server."""
        return [Reference(x) for x in self.__make_grpc_call("AvailableCheckpoints", Empty(), method=b"list", signing_keys=signing_keys).list]

    def get_available_devices(self, *, signing_keys: List[SigningKey] = []) -> List[str]:
        """Returns the list of devices available on the server."""
        return self.__make_grpc_call("AvailableDevices", Empty(), signing_keys=signing_keys).list

    def get_available_optimizers(self, *, signing_keys: List[SigningKey] = []) -> List[str]:
        """Returns the list of optimizers supported by the server."""
        return self.__make_grpc_call("AvailableOptimizers", Empty(), signing_keys=signing_keys).list

    def train(self, config: TrainRequest, *, signing_keys: List[SigningKey] = []) -> Reference:
        """Trains a model with hyperparameters defined in `config` on the BastionAI server.

        Args:
            config: Training configuration that specifies the model, dataset and hyperparameters.
        """
        return Reference(self.__make_grpc_call("Train", config, method=b"train", signing_keys=signing_keys))

    def test(self, config: TestRequest, *, signing_keys: List[SigningKey] = []) -> Reference:
        """Tests a dataset on a model according to `config` on the BastionAI server.

        Args:
            config: Testing configuration that specifies the model, dataset and hyperparameters.
        """
        return Reference(self.__make_grpc_call("Test", config, method=b"test", signing_keys=signing_keys))

    def delete_dataset(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> None:
        """Deletes the dataset correponding to the given `ref` reference on the BastionAI server.

        Args:
            ref: BastionAI gRPC protocol reference of the dataset to be deleted.
        """
        self.__make_grpc_call("DeleteDataset", ref.request(), method=b"delete", signing_keys=signing_keys)

    def delete_module(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> None:
        """Deletes the module correponding to the given `ref` reference on the BastionAI server.

        Args:
            ref: BastionAI gRPC protocol reference of the module to be deleted.
        """
        self.__make_grpc_call("DeleteModule", ref.request(), method=b"delete", signing_keys=signing_keys)
    
    def delete_checkpoint(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> None:
        self.__make_grpc_call("DeleteCheckpoint", ref.request(), method=b"delete", signing_keys=signing_keys)

    def delete_run(self, ref: Reference, *, signing_keys: List[SigningKey] = []) -> None:
        self.__make_grpc_call("DeleteRun", ref.request(), method=b"delete", signing_keys=signing_keys)

    def get_metric(self, run: Reference, *, signing_keys: List[SigningKey] = []) -> MetricResponse:
        """Returns the value of the metric associated with the given `run` reference.

        Args:
            run: BastionAI gRPC protocol reference of the run whose metric is read.
        """
        return self.__make_grpc_call("GetMetric", run.request(), method=b"fetch", signing_keys=signing_keys)
    
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


# @dataclass
class Connection:
    """Context manger that handles a connection to a BastionAI server.
    It returns a `Client` to use the connexion within its context.
    
    Args:
        host: Hostname of the BastionAI server.
        port: Port of the BastionAI server.
        default_license: Default owner license passed to the constructor of the return `Client`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        default_license: Optional[Union[LicenseBuilder, License]] = None,
        channel: Any = None,
        server_name: str = "bastionai-srv",
        license_key = Optional[SigningKey],
        default_signing_keys: List[SigningKey] = []
    ):
        self.host = host
        self.port = port
        self.default_license = default_license
        self.channel = channel
        self.server_name = server_name
        self.license_key = license_key
        self.default_signing_keys = default_signing_keys

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
            # certificate_chain=cert_chain, private_key=cert_key,
            root_certificates=bytes(server_cert, encoding="utf8")
        )

        server_target = f"{self.host}:{self.port}"
        self.channel = grpc.secure_channel(
            server_target, server_cred, options=connection_options
        )

        if self.license_key is not None:
            if self.default_license is None:
                self.default_license = LicenseBuilder.default_with_pubkey(self.license_key.pubkey).build()
            if self.license_key not in self.default_signing_keys:
                self.default_signing_keys = [*self.default_signing_keys, self.license_key]

        return Client(
            RemoteTorchStub(self.channel),
            client_info=client_info,
            default_license=self.default_license,
            default_signing_keys=self.default_signing_keys,
        )

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        self.channel.close()
