from typing import Optional, Union, List
from time import sleep
from tqdm import tqdm  # type: ignore [import]
from grpc import StatusCode
from torch.nn import Module
from torch.utils.data import Dataset
import torch
from ..pb.bastionlab_torch_pb2 import Metric, Reference, TestConfig, TrainConfig  # type: ignore [import]
from ..errors import GRPCException
from .psg import expand_weights
from .client import BastionLabTorch
from .optimizer_config import *

from .utils import bulk_deserialize


class RemoteDataset:
    """Represents a remote dataloader on the BlindAI server encapsulating a training
    and optional testing datasets along with dataloading parameters.

    Args:
        client: A BastionAI client to be used to access server resources.
        train_dataset: A `torch.utils.data.Dataset` instance for training that will be uploaded on the server.
        test_dataset: An optional `torch.utils.data.Dataset` instance for testing that will be uploaded on the server.
        name: A name for the uploaded dataset.
        description: A string description of the dataset being uploaded.
    """

    def __init__(
        self,
        client: BastionLabTorch,
        train_dataset: Union[Dataset, Reference],
        test_dataset: Optional[Union[Dataset, Reference]] = None,
        privacy_limit: Optional[float] = None,
        name: Optional[str] = None,
        description: str = "",
        progress: bool = True,
    ) -> None:
        if isinstance(train_dataset, Dataset):
            self.train_dataset_ref = client.send_dataset(
                train_dataset,
                name=name if name is not None else type(train_dataset).__name__,
                description=description,
                privacy_limit=privacy_limit,
                progress=progress,
            )
            self.name = name
            self.description = description
            self.trace_input = [input.unsqueeze(0) for input in train_dataset[0][0]]
            self.nb_samples = len(train_dataset)  # type: ignore [arg-type]
            self.privacy_limit = privacy_limit
        else:
            self.train_dataset_ref = train_dataset
            self.name = self.train_dataset_ref.name
            self.description = self.train_dataset_ref.description
            meta = bulk_deserialize(train_dataset.meta)
            self.trace_input = [
                torch.zeros(s, dtype=dtype)
                if dtype
                in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
                else torch.randn(s, dtype=dtype)
                for s, dtype in zip(meta["input_shape"], meta["input_dtype"])
            ]
            self.nb_samples = meta["nb_samples"]
            self.privacy_limit = meta["privacy_limit"]
        if test_dataset is not None and isinstance(test_dataset, Dataset):
            self.test_dataset_ref = client.send_dataset(
                test_dataset,
                name=f"{name} (test)"
                if name is not None
                else type(test_dataset).__name__,
                description=description,
                privacy_limit=privacy_limit,
                train_dataset=self.train_dataset_ref,
                progress=progress,
            )
        else:
            self.test_dataset_ref = test_dataset
        self.client = client

    @staticmethod
    def list_available(client: BastionLabTorch) -> List["RemoteDataset"]:
        """Returns the list of `RemoteDataset`s available on the server."""
        refs = client.get_available_datasets()
        ds = [(ref, bulk_deserialize(ref.meta)["train_dataset"]) for ref in refs]
        return [RemoteDataset(client, d[1], d[0]) for d in ds if d[1] is not None]

    def __str__(self) -> str:
        return f"{self.name} ({self.train_dataset_ref.identifier}): size={self.nb_samples}, desc={self.description if len(self.description) > 0 else 'N/A'}"

    def __format__(self, __format_spec: str) -> str:
        return self.__str__()

    def _set_test_dataset(
        self, test_dataset: Union[Dataset, Reference], progress: bool = True
    ) -> None:
        if not type(test_dataset) == Reference:
            self.test_dataset_ref = self.client.send_dataset(
                test_dataset,
                name=f"{self.name} (test)"
                if self.description is not None
                else type(test_dataset).__name__,
                description=self.description,
                privacy_limit=self.privacy_limit,
                train_dataset=self.train_dataset_ref,
                progress=progress,
            )
        else:
            self.test_dataset_ref = test_dataset


class RemoteLearner:
    """Represents a remote model on the server along with hyperparameters to train and test it.

    The remote learner accepts the model to be trained with a `RemoteDataLoader`.

    Args:
        client: A BastionAI client to be used to access server resources.
        model: A Pytorch nn.Module or a BastionAI gRPC protocol reference to a distant model.
        remote_dataset: A BastionAI remote dataloader.
        loss: The name of the loss to use for training the model, supported loss functions are "l2" and "cross_entropy".
        optimizer: The configuration of the optimizer to use during training, refer to the documentation of `OptimizerConfig`.
        device: Name of the device on which to train model. The list of supported devices may be obtained using the
                `get_available_devices` endpoint of the `BastionLabTorch` object.
        max_grad_norm: This specifies the clipping threshold for gradients in DP-SGD.
        metric_eps_per_batch: The privacy budget allocated to the disclosure of the loss of every batch.
                              May be overriden by providing a global budget for the loss disclosure over the whole training
                              on calling the `fit` method.
        model_name: A name for the uploaded model.
        model_description: Provides additional description for the uploaded model.
        expand: Whether to expand model's weights prior to uploading it, or not.
        progress: Whether to display a tqdm progress bar or not.
    """

    def __init__(
        self,
        client: BastionLabTorch,
        model: Union[Module, Reference],
        remote_dataset: Union[RemoteDataset, Reference],
        loss: str,
        max_batch_size: int,
        optimizer: OptimizerConfig = Adam(),
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        metric_eps_per_batch: Optional[float] = None,
        model_name: Optional[str] = None,
        model_description: str = "",
        expand: bool = True,
        progress: bool = True,
    ) -> None:
        if isinstance(model, Module):
            model_class_name = type(model).__name__

            if expand:
                expand_weights(model, max_batch_size)
            self.model = model
            try:
                model = torch.jit.script(model)
            except:
                model = torch.jit.trace(  # Compile the model with the tracing strategy
                    # Wrapp the model to use the first output only (and drop the others)
                    model,
                    [x.unsqueeze(0) for x in remote_dataset.trace_input],
                )
            self.model_ref = client.send_model(
                model,
                name=model_name if model_name is not None else model_class_name,
                description=model_description,
                progress=True,
            )
        else:
            self.model_ref = model
        self.remote_dataset = (
            remote_dataset
            if type(remote_dataset) == RemoteDataset
            else RemoteDataset(client, remote_dataset)
        )
        self.client = client
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_grad_norm = max_grad_norm
        self.metric_eps_per_batch = (
            0.01
            if self.remote_dataset.privacy_limit is not None
            and metric_eps_per_batch is None
            else (metric_eps_per_batch if metric_eps_per_batch is not None else -1.0)
        )
        self.progress = progress
        self.log: List[Metric] = []

    def _train_config(
        self,
        nb_epochs: int,
        eps: Optional[float],
        batch_size: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        metric_eps: Optional[float] = None,
        per_n_epochs_checkpoint: int = 0,
        per_n_steps_checkpoint: int = 0,
        resume: bool = False,
    ) -> TrainConfig:
        batch_size = batch_size if batch_size is not None else self.max_batch_size
        return TrainConfig(
            model=self.model_ref,
            dataset=self.remote_dataset.train_dataset_ref,
            batch_size=batch_size,
            epochs=nb_epochs,
            device=self.device,
            metric=self.loss,
            per_n_steps_checkpoint=per_n_steps_checkpoint,
            per_n_epochs_checkpoint=per_n_epochs_checkpoint,
            resume=resume,
            eps=eps if eps is not None else -1.0,
            max_grad_norm=max_grad_norm if max_grad_norm else self.max_grad_norm,
            metric_eps=metric_eps
            if metric_eps
            else self.metric_eps_per_batch
            * float(nb_epochs)
            * float(self.remote_dataset.nb_samples / batch_size),
            **self.optimizer.to_msg_dict(lr),
        )

    def _test_config(
        self,
        batch_size: Optional[int] = None,
        metric: Optional[str] = None,
        metric_eps: Optional[float] = None,
    ) -> TestConfig:
        batch_size = batch_size if batch_size is not None else self.max_batch_size
        return TestConfig(
            model=self.model_ref,
            dataset=self.remote_dataset.test_dataset_ref,
            batch_size=batch_size,
            device=self.device,
            metric=metric if metric is not None else self.loss,
            metric_eps=metric_eps
            if metric_eps
            else self.metric_eps_per_batch
            * float(self.remote_dataset.nb_samples / batch_size),
        )

    @staticmethod
    def _new_tqdm_bar(
        epoch: int, nb_epochs: int, nb_batches: int, train: bool = True
    ) -> tqdm:
        t = tqdm(
            total=nb_batches,
            unit="batch",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        t.set_description(
            "Epoch {}/{} - {}".format(epoch, nb_epochs, "train" if train else "test")
        )
        return t

    def _poll_metric(
        self,
        run: Reference,
        name: str,
        train: bool = True,
        timeout: int = 100,
        poll_delay: float = 0.2,
    ) -> None:
        timeout_counter = 0

        metric = None
        for _ in range(timeout):
            try:
                sleep(poll_delay)
                metric = self.client.get_metric(run)
                break
            except GRPCException as e:
                if e.code == StatusCode.OUT_OF_RANGE:
                    continue
                else:
                    raise e
        if metric is None:
            raise Exception(
                f"Run start timeout. Polling has stoped. You may query the server by hand later using: run id is {run.identifier}"
            )

        if self.progress:
            t = RemoteLearner._new_tqdm_bar(
                metric.epoch + 1, metric.nb_epochs, metric.nb_batches, train
            )
            t.update(metric.batch + 1)
            t.set_postfix(
                **{name: "{:.4f} (+/- {:.4f})".format(metric.value, metric.uncertainty)}
            )
        else:
            self.log.append(metric)

        while True:
            sleep(poll_delay)
            prev_batch = metric.batch
            prev_epoch = metric.epoch
            metric = self.client.get_metric(run)

            # Handle end of training
            if metric.batch == prev_batch and metric.epoch == prev_epoch:
                timeout_counter += 1
            else:
                timeout_counter = 0
            if timeout_counter > timeout:
                break

            if self.progress:
                # Handle bar update
                if metric.epoch != prev_epoch:
                    t = RemoteLearner._new_tqdm_bar(
                        metric.epoch + 1, metric.nb_epochs, metric.nb_batches, train
                    )
                    t.update(metric.batch + 1)
                else:
                    t.update(metric.batch - prev_batch)
                t.set_postfix(
                    **{
                        name: "{:.4f} (+/- {:.4f})".format(
                            metric.value, metric.uncertainty
                        )
                    }
                )
            else:
                self.log.append(metric)

            if (
                metric.epoch + 1 == metric.nb_epochs
                and metric.batch + 1 == metric.nb_batches
            ):
                break

    def fit(
        self,
        nb_epochs: int,
        eps: Optional[float],
        batch_size: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        metric_eps: Optional[float] = None,
        timeout: float = 60.0,
        poll_delay: float = 0.2,
        per_n_epochs_checkpoint: int = 0,
        per_n_steps_checkpoint: int = 0,
        resume: bool = False,
    ) -> None:
        """Fits the uploaded model to the training dataset with given hyperparameters.

        Args:
            nb_epocs: Specifies the number of epochs to train the model.
            eps: Specifies the global privacy budget for the DP-SGD algorithm.
            max_grad_norm: Overrides the default clipping threshold for gradients passed to the constructor.
            lr: Overrides the default learning rate of the optimizer config passed to the constructor.
            metric_eps: Global privacy budget for loss disclosure for the whole training that overrides
                        the default per-batch budget.
            timeout: Timeout in seconds between two updates of the loss on the server side. When elapsed without updates,
                     polling ends and the progress bar is terminated.
            poll_delay: Delay in seconds between two polling requests for the loss.
        """
        run = self.client.train(
            self._train_config(
                nb_epochs,
                eps,
                batch_size,
                max_grad_norm,
                lr,
                metric_eps,
                per_n_epochs_checkpoint,
                per_n_steps_checkpoint,
                resume,
            )
        )
        self._poll_metric(
            run,
            name=self.loss,
            train=True,
            timeout=int(timeout / poll_delay),
            poll_delay=poll_delay,
        )

    def test(
        self,
        test_dataset: Optional[Union[Dataset, Reference]] = None,
        batch_size: Optional[int] = None,
        metric: Optional[str] = None,
        metric_eps: Optional[float] = None,
        timeout: int = 100,
        poll_delay: float = 0.2,
    ) -> None:
        """Tests the remote model with the test dataloader provided in the RemoteDataLoader.

        Args:
            test_dataset: overrides the test dataset passed to the remote `RemoteDataset` constructor.
            metric: test metric name, if not providedm the training loss is used. Metrics available are loss functions and `accuracy`.
            metric_eps: Global privacy budget for metric disclosure for the whole testing procedure that overrides
                        the default per-batch budget.
            timeout: Timeout in seconds between two updates of the metric on the server side. When elapsed without updates,
                     polling ends and the progress bar is terminated.
            poll_delay: Delay in seconds between two polling requests for the metric.
        """
        if test_dataset is not None:
            self.remote_dataset._set_test_dataset(test_dataset)
        run = self.client.test(self._test_config(batch_size, metric, metric_eps))
        self._poll_metric(
            run,
            name=metric if metric is not None else self.loss,
            train=False,
            timeout=timeout,
            poll_delay=poll_delay,
        )

    def get_model(self) -> Module:
        """Returns the model passed to the constructor with its weights
        updated with the weights obtained by training on the server.
        """
        self.client.fetch_model_weights(self.model, self.model_ref)
        return self.model
