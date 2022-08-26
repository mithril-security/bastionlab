from typing import Optional, Union

from bastionai.pb.remote_torch_pb2 import Reference, TestConfig, TrainConfig  # type: ignore [import]
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
import torch
from bastionai.psg import expand_weights
from bastionai.client import Client
from bastionai.optimizer_config import *


class RemoteDataLoader:
    def __init__(
        self,
        client: Client,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        description: Optional[str] = None,
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
        if train_dataloader.batch_size != test_dataloader.batch_size:
            raise Exception("Train and test dataloaders must use the same batch size.")
        self.train_dataset_ref = client.send_dataset(
            train_dataloader.dataset,
            description
            if description is not None
            else type(train_dataloader.dataset).__name__,
            secret if secret is not None else client.default_secret,
        )
        self.test_dataset_ref = client.send_dataset(
            test_dataloader.dataset,
            f"[Test set] {description}"
            if description is not None
            else type(test_dataloader.dataset).__name__,
            secret if secret is not None else client.default_secret,
        )
        self.trace_input, _ = train_dataloader.dataset[0]
        self.client = client
        if train_dataloader.batch_size is None:
            raise Exception("A batch size must be provided to the dataloader.")
        self.batch_size: int = train_dataloader.batch_size
        self.nb_samples = len(train_dataloader.dataset)  # type: ignore [arg-type]


class RemoteLearner:
    def __init__(
        self,
        client: Client,
        model: Union[Module, Reference],
        remote_dataloader: RemoteDataLoader,
        metric: str,
        optimizer: OptimizerConfig = Adam(),
        device: str = "cpu",
        max_grad_norm: float = 4.0,
        model_description: Optional[str] = None,
        secret: Optional[bytes] = None,
        expand: bool = True,
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
                model_description
                if model_description is not None
                else model_class_name,
                secret if secret is not None else client.default_secret,
            )
        else:
            self.model_ref = model
        self.remote_dataloader = remote_dataloader
        self.client = client
        self.metric = metric
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm

    def _dp_config(
        self,
        nb_epochs: int,
        eps: float,
        max_grad_norm: Optional[float] = None,
    ) -> TrainConfig.DpParameters:
        delta = 1 / (10 * self.remote_dataloader.nb_samples)
        q = self.remote_dataloader.batch_size / self.remote_dataloader.nb_samples
        t = float(nb_epochs) / q
        max_grad_norm = (
            max_grad_norm if max_grad_norm is not None else self.max_grad_norm
        )
        return TrainConfig.DpParameters(
            max_grad_norm=max_grad_norm,
            noise_multiplier=np.sqrt(2 * np.log(1.25 / delta)) * q * np.sqrt(t) / eps,
        )

    def _train_config(
        self,
        nb_epochs: int,
        eps: float,
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> TrainConfig:
        return TrainConfig(
            model=self.model_ref,
            dataset=self.remote_dataloader.train_dataset_ref,
            batch_size=self.remote_dataloader.batch_size,
            epochs=nb_epochs,
            device=self.device,
            metric=self.metric,
            differential_privacy=self._dp_config(nb_epochs, eps, max_grad_norm),
            **self.optimizer.to_msg_dict(lr),
        )

    def _test_config(self, metric: Optional[str] = None) -> TestConfig:
        return TestConfig(
            model=self.model_ref,
            dataset=self.remote_dataloader.test_dataset_ref,
            batch_size=self.remote_dataloader.batch_size,
            device=self.device,
            metric=metric if metric is not None else self.metric,
        )

    def fit(
        self,
        nb_epochs: int,
        eps: float,
        max_grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Fit an uploaded model to the provided parameters.

        Args:
            nb_epocs (int): Specifies the number of epochs to fit the model.
            eps (float): Specifies the epsilon for differential privacy step.
            max_grad_norm (Optional[float], optional): Specifies the clipping threshold for gradients in DP-SGD. Defaults to None.
            lr (Optional[float], optional): Specifies the learning rate. Defaults to None.
        """
        self.client.train(self._train_config(nb_epochs, eps, max_grad_norm, lr))

    def test(self, metric: Optional[str] = None) -> None:
        """Tests the remote model with the test dataloader provided in the RemoteDataLoader.

        Args:
            metric (Optional[str], optional): Specifies the preferred loss metric. Defaults to None.
        """
        self.client.test(self._test_config(metric))

    def get_model(self) -> Module:
        """Retrieves the trained model from BastionAI

        Returns:
            torch.nn.Module:
                A Pytorch module.
        """
        self.client.fetch_model_weights(self.model, self.model_ref)
        return self.model
