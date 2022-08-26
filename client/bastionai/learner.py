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

from bastionai.utils import PrivacyBudget, Private, NotPrivate


class RemoteDataLoader:
    def __init__(
        self,
        client: Client,
        train_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        privacy_limit: PrivacyBudget = NotPrivate(),
        name: Optional[str] = None,
        description: str = "",
        secret: Optional[bytes] = None,
    ) -> None:
        """RemoteDataLoader class creates a remote dataloader on BastionAI with the training and testing datasets

        Args:
            client (Client): A BastionAI client connection
            train_dataloader (DataLoader): Dataloader serving the training dataset.
            test_dataloader (DataLoader): Dataloader serving the testing dataset.
            description (Optional[str], optional): A string description of the dataset being uploaded. Defaults to None.
            secret (Optional[bytes], optional): User secret to secure training and testing datasets with. Defaults to None.
        """
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
    def __init__(
        self,
        client: Client,
        model: Union[Module, Reference],
        remote_dataloader: RemoteDataLoader,
        metric: str,
        optimizer: OptimizerConfig = Adam(),
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        metric_eps_per_batch: PrivacyBudget = Private(1.0),
        model_name: Optional[str] = None,
        model_description: str = "",
        secret: Optional[bytes] = None,
        expand: bool = True,
        progress: bool = True,
    ) -> None:
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
        self.metric = metric
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.metric_eps = metric_eps_per_batch
        self.progress = progress
        self.log: List[Metric] = []

    def _train_config(
        self,
        nb_epochs: int,
        eps: PrivacyBudget,
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        metric_eps: Optional[PrivacyBudget] = None,
    ) -> TrainConfig:
        return TrainConfig(
            model=self.model_ref,
            dataset=self.remote_dataloader.train_dataset_ref,
            batch_size=self.remote_dataloader.batch_size,
            epochs=nb_epochs,
            device=self.device,
            metric=self.metric,
            eps=eps.into_float(),
            max_grad_norm=max_grad_norm if max_grad_norm else self.max_grad_norm,
            metric_eps=metric_eps.into_float()
            if metric_eps
            else self.metric_eps.into_float()
            * float(nb_epochs)
            * float(
                self.remote_dataloader.nb_samples / self.remote_dataloader.batch_size
            ),
            **self.optimizer.to_msg_dict(lr),
        )

    def _test_config(
        self, metric: Optional[str] = None, metric_eps: Optional[PrivacyBudget] = None
    ) -> TestConfig:
        return TestConfig(
            model=self.model_ref,
            dataset=self.remote_dataloader.test_dataset_ref,
            batch_size=self.remote_dataloader.batch_size,
            device=self.device,
            metric=metric if metric is not None else self.metric,
            metric_eps=metric_eps.into_float()
            if metric_eps
            else self.metric_eps.into_float()
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
        train: bool = True,
        timeout: int = 100,
        poll_delay: float = 0.2,
    ) -> None:
        name = self.metric
        timeout_counter = 0

        metric = None
        for _ in range(timeout):
            try:
                sleep(poll_delay)
                metric = self.client.get_metric(run)
                print("Poll")
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
        eps: PrivacyBudget,
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        metric_eps: Optional[PrivacyBudget] = None,
        timeout: int = 100,
        poll_delay: float = 0.2,
    ) -> None:
        """Fit an uploaded model to the provided parameters.

        Args:
            nb_epocs (int): Specifies the number of epochs to fit the model.
            eps (float): Specifies the epsilon for differential privacy step.
            max_grad_norm (Optional[float], optional): Specifies the clipping threshold for gradients in DP-SGD. Defaults to None.
            lr (Optional[float], optional): Specifies the learning rate. Defaults to None.
        """
        self.client.train(self._train_config(nb_epochs, eps, max_grad_norm, lr))

    def test(
        self,
        test_dataloader: Optional[DataLoader] = None,
        metric: Optional[str] = None,
        metric_eps: Optional[PrivacyBudget] = None,
        timeout: int = 100,
        poll_delay: float = 0.2,
    ) -> None:
        """Tests the remote model with the test dataloader provided in the RemoteDataLoader.

        Args:
            metric (Optional[str], optional): Specifies the preferred loss metric. Defaults to None.
        """
        if test_dataloader is not None:
            self.remote_dataloader._set_test_dataloader(test_dataloader)
        run = self.client.test(self._test_config(metric, metric_eps))
        self._poll_metric(run, train=False, timeout=timeout, poll_delay=poll_delay)

    def get_model(self) -> Module:
        """Retrieves the trained model from BastionAI

        Returns:
            torch.nn.Module:
                A Pytorch module.
        """
        self.client.fetch_model_weights(self.model, self.model_ref)
        return self.model
