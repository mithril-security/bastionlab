import grpc
import polars as pl
from ..pb.bastionlab_linfa_pb2_grpc import LinfaServiceStub
from ..pb.bastionlab_linfa_pb2 import (
    TrainingRequest,
    PredictionRequest,
    ValidationRequest,
)
from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from ..polars import FetchableLazyFrame, BastionLabPolars
    from .trainers import Trainer
    from .remote_linfa import FittedModel
    from ..client import Client


class BastionLabLinfa:
    def __init__(
        self,
        client: "Client",
        polars: "BastionLabPolars",
    ) -> None:
        self.stub = LinfaServiceStub(client._channel)
        self.polars = polars

    def train(
        self,
        records: "FetchableLazyFrame",
        target: "FetchableLazyFrame",
        trainer: "Trainer",
        ratio: float = 1.0,
    ) -> "FittedModel":
        from .remote_linfa import FittedModel

        res = self.stub.Train(
            TrainingRequest(
                records=records.identifier,
                target=target.identifier,
                ratio=ratio,
                **trainer.to_msg_dict(),
            )
        )
        return FittedModel._from_reference(res, trainer)

    def predict(self, model: "FittedModel", data: List[float]) -> pl.DataFrame:
        from ..polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(model=model.identifier, data=data, probability=False)
        )
        return FetchableLazyFrame._from_reference(self.polars, res).fetch()

    def predict_proba(
        self, model: "FittedModel", data: List[float]
    ) -> "FetchableLazyFrame":
        from ..polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(model=model.identifier, data=data, probability=True)
        )
        return FetchableLazyFrame._from_reference(self.polars, res).fetch()

    def cross_validate(
        self,
        model: "FittedModel",
    ) -> pl.DataFrame:
        from ..polars import FetchableLazyFrame

        res = self.stub.CrossValidate(ValidationRequest(model=model.identifier))
        return FetchableLazyFrame._from_reference(self.polars, res).fetch()