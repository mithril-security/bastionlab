from typing import Optional, Union, List

from bastionai.pb.remote_torch_pb2 import Metric, Reference, TestConfig, TrainConfig  # type: ignore [import]
from torch.nn import Module
from torch.utils.data import DataLoader
import torch
from bastionai.psg import expand_weights
from bastionai.client import Client
from bastionai.optimizer_config import *

from time import sleep
from tqdm import tqdm  # type: ignore [import]


class RemoteDataLoader:
    """Represents a remote dataloader on the BlindAI server encapsulating a training
    and optional testing datasets along with dataloading parameters.

    Args:
        client: A BastionAI client to be used to access server resources.
        train_dataloader: A `torch.utils.data.DataLoader` instance for training that will be uploaded on the server.
        test_dataloader: An optional `torch.utils.data.DataLoader` instance for testing that will be uploaded on the server.
        name: A name for the uploaded dataset.
        description: A string description of the dataset being uploaded.
        secret: [In progress] Owner secret override for the uploaded data.
    """
    def __init__(
        self,
        client: Client,
        train_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        privacy_limit: Optional[float] = None,
        name: Optional[str] = None,
        description: str = "",
        secret: Optional[bytes] = None,
    ) -> None:
        if (
            test_dataloader is not None
            and train_dataloader.batch_size != test_dataloader.batch_size
        ):
            raise Exception("Train and test dataloaders must use the same batch size.")
        self.train_dataset_ref = client.send_dataset(
            train_dataloader.dataset,
            name=name if name is not None else type(train_dataloader.dataset).__name__,
            description=description,
            secret=secret,
            privacy_limit=privacy_limit,
        )
        if test_dataloader is not None:
            self.test_dataset_ref = client.send_dataset(
                test_dataloader.dataset,
                name=f"{name} (test)"
                if name is not None
                else type(test_dataloader.dataset).__name__,
                description=description,
                secret=secret,
                privacy_limit=privacy_limit,
            )
        else:
            self.test_dataset_ref = None
        self.trace_input, _ = train_dataloader.dataset[0]
        self.client = client
        if train_dataloader.batch_size is None:
            raise Exception("A batch size must be provided to the dataloader.")
        self.batch_size: int = train_dataloader.batch_size
        self.name = name
        self.description = description
        self.secret = secret
        self.privacy_limit = privacy_limit
        self.nb_samples = len(train_dataloader.dataset)  # type: ignore [arg-type]

    def _set_test_dataloader(self, test_dataloader: DataLoader) -> None:
        if self.batch_size != test_dataloader.batch_size:
            raise Exception("Train and test dataloaders must use the same batch size.")
        self.test_dataset_ref = self.client.send_dataset(
            test_dataloader.dataset,
            name=f"{self.name} (test)"
            if self.description is not None
            else type(test_dataloader.dataset).__name__,
            description=self.description,
            secret=self.secret,
            privacy_limit=self.privacy_limit,
        )


class RemoteLearner:
    """Represents a remote model on the server along with hyperparameters to train and test it.
    
    The remote learner accepts the model to be trained with a `RemoteDataLoader`.

    Args:
        client: A BastionAI client to be used to access server resources.
        model: A Pytorch nn.Module or a BastionAI gRPC protocol reference to a distant model.
        remote_dataloader: A BastionAI remote dataloader.
        loss: The name of the loss to use for training the model, supported loss functions are "l2" and "cross_entropy".
        optimizer: The configuration of the optimizer to use during training, refer to the documentation of `OptimizerConfig`.
        device: Name of the device on which to train model. The list of supported devices may be obtained using the
                `get_available_devices` endpoint of the `Client` object.
        max_grad_norm: This specifies the clipping threshold for gradients in DP-SGD.
        metric_eps_per_batch: The privacy budget allocated to the disclosure of the loss of every batch.
                              May be overriden by providing a global budget for the loss disclosure over the whole training
                              on calling the `fit` method.
        model_name: A name for the uploaded model.
        model_description: Provides additional description for the uploaded model.
        secret: [In progress] Owner secret override for the uploaded model.
        expand: Whether to expand model's weights prior to uploading it, or not.
        progress: Whether to display a tqdm progress bar or not.
    """
    def __init__(
        self,
        client: Client,
        model: Union[Module, Reference],
        remote_dataloader: RemoteDataLoader,
        loss: str,
        optimizer: OptimizerConfig = Adam(),
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        metric_eps_per_batch: Optional[float] = None,
        model_name: Optional[str] = None,
        model_description: str = "",
        secret: Optional[bytes] = None,
        expand: bool = True,
        progress: bool = True,
    ) -> None:
        if isinstance(model, Module):
            model_class_name = type(model).__name__

            if expand:
                expand_weights(model, remote_dataloader.batch_size)
            self.model = model
            try:
                model = torch.jit.script(model)
            except:
                model = torch.jit.trace(  # Compile the model with the tracing strategy
                    # Wrapp the model to use the first output only (and drop the others)
                    model,
                    [x.unsqueeze(0) for x in remote_dataloader.trace_input],
                )
            self.model_ref = client.send_model(
                model,
                name=model_name if model_name is not None else model_class_name,
                description=model_description,
                secret=secret,
            )
        else:
            self.model_ref = model
        self.remote_dataloader = remote_dataloader
        self.client = client
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.metric_eps_per_batch = (
            0.01
            if self.remote_dataloader.privacy_limit is not None
            and metric_eps_per_batch is None
            else (metric_eps_per_batch if metric_eps_per_batch is not None else -1.0)
        )
        self.progress = progress
        self.log: List[Metric] = []

    def _train_config(
        self,
        nb_epochs: int,
        eps: Optional[float],
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        metric_eps: Optional[float] = None,
    ) -> TrainConfig:
        return TrainConfig(
            model=self.model_ref,
            dataset=self.remote_dataloader.train_dataset_ref,
            batch_size=self.remote_dataloader.batch_size,
            epochs=nb_epochs,
            device=self.device,
            metric=self.loss,
            eps=eps if eps is not None else -1.0,
            max_grad_norm=max_grad_norm if max_grad_norm else self.max_grad_norm,
            metric_eps=metric_eps
            if metric_eps
            else self.metric_eps_per_batch
            * float(nb_epochs)
            * float(
                self.remote_dataloader.nb_samples / self.remote_dataloader.batch_size
            ),
            **self.optimizer.to_msg_dict(lr),
        )

    def _test_config(
        self, metric: Optional[str] = None, metric_eps: Optional[float] = None
    ) -> TestConfig:
        return TestConfig(
            model=self.model_ref,
            dataset=self.remote_dataloader.test_dataset_ref,
            batch_size=self.remote_dataloader.batch_size,
            device=self.device,
            metric=metric if metric is not None else self.loss,
            metric_eps=metric_eps
            if metric_eps
            else self.metric_eps_per_batch
            * float(
                self.remote_dataloader.nb_samples / self.remote_dataloader.batch_size
            ),
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
            except:
                continue
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
            if (
                metric.epoch + 1 == metric.nb_epochs
                and metric.batch + 1 == metric.nb_batches
            ):
                break
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

    def fit(
        self,
        nb_epochs: int,
        eps: Optional[float],
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        metric_eps: Optional[float] = None,
        timeout: float = 60.0,
        poll_delay: float = 0.2,
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
            self._train_config(nb_epochs, eps, max_grad_norm, lr, metric_eps)
        )
        self._poll_metric(
            run, name=self.loss, train=True, timeout=int(timeout / poll_delay), poll_delay=poll_delay
        )

    def test(
        self,
        test_dataloader: Optional[DataLoader] = None,
        metric: Optional[str] = None,
        metric_eps: Optional[float] = None,
        timeout: int = 100,
        poll_delay: float = 0.2,
    ) -> None:
        """Tests the remote model with the test dataloader provided in the RemoteDataLoader.

        Args:
            test_dataloader: overrides the test dataloader passed to the remote `RemoteDataLoader` constructor.
            metric: test metric name, if not providedm the training loss is used. Metrics available are loss functions and `accuracy`.
            metric_eps: Global privacy budget for metric disclosure for the whole testing procedure that overrides
                        the default per-batch budget.
            timeout: Timeout in seconds between two updates of the metric on the server side. When elapsed without updates,
                     polling ends and the progress bar is terminated.
            poll_delay: Delay in seconds between two polling requests for the metric.
        """
        if test_dataloader is not None:
            self.remote_dataloader._set_test_dataloader(test_dataloader)
        run = self.client.test(self._test_config(metric, metric_eps))
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
