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
    from ..polars.remote_polars import RemoteArray, BastionLabPolars, FetchableLazyFrame
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

    def _train(
        self,
        records: "RemoteArray",
        target: "RemoteArray",
        trainer: "Trainer",
    ) -> "FittedModel":
        from .remote_linfa import FittedModel

        res = self.stub.Train(
            TrainingRequest(
                records=records.identifier,
                target=target.identifier,
                **trainer.to_msg_dict(),
            )
        )
        return FittedModel._from_reference(res, trainer)

    def _predict(self, model: "FittedModel", test_set: "RemoteArray") -> pl.DataFrame:
        from ..polars.remote_polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(
                model=model.identifier, test_set=test_set.identifier, probability=False
            )
        )
        return FetchableLazyFrame._from_reference(self.polars, res).fetch()

    def predict_proba(
        self, model: "FittedModel", test_set: "RemoteArray"
    ) -> "FetchableLazyFrame":
        from ..polars.remote_polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(
                model=model.identifier, data=test_set.identifier, probability=True
            )
        )
        return FetchableLazyFrame._from_reference(self.polars, res).fetch()

    def cross_validate(
        self,
        model: "FittedModel",
    ) -> pl.DataFrame:
        from ..polars.remote_polars import FetchableLazyFrame

        res = self.stub.CrossValidate(ValidationRequest(model=model.identifier))
        return FetchableLazyFrame._from_reference(self.polars, res).fetch()
