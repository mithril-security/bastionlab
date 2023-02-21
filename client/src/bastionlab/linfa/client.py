import grpc
import polars as pl
from ..pb.bastionlab_linfa_pb2_grpc import LinfaServiceStub
from ..pb.bastionlab_linfa_pb2 import (
    TrainingRequest,
    PredictionRequest,
    ValidationRequest,
    Trainer as LinfaTrainer,
)
from ..pb.bastionlab_polars_pb2 import ReferenceResponse
from typing import TYPE_CHECKING, List
from ..config import CONFIG

if TYPE_CHECKING:
    from ..polars.remote_polars import RemoteArray, FetchableLazyFrame
    from .remote_linfa import FittedModel, Trainer
    from ..client import Client


class BastionLabLinfa:
    def __init__(
        self,
        client: "Client",
    ) -> None:
        self.stub = LinfaServiceStub(client._channel)
        self.client = client

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
                trainer=LinfaTrainer(**trainer.to_msg_dict()),
            )
        )
        return FittedModel._from_reference(res, trainer._name)

    def _predict(
        self, model: "FittedModel", pred_input: "RemoteArray"
    ) -> "FetchableLazyFrame":
        from ..polars.remote_polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(
                model=model.identifier, input=pred_input.identifier, probability=False
            )
        )
        return FetchableLazyFrame._from_reference(self.client.polars, res)

    def _predict_proba(
        self, model: "FittedModel", pred_input: "RemoteArray"
    ) -> "FetchableLazyFrame":
        from ..polars.remote_polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(
                model=model.identifier, input=pred_input.identifier, probability=True
            )
        )
        return FetchableLazyFrame._from_reference(self.client.polars, res)


def cross_validate(
    trainer: "Trainer", X: "RemoteArray", y: "RemoteArray", cv: int, scoring: str = "r2"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = X._client.linfa.stub.CrossValidate(
        ValidationRequest(
            trainer=LinfaTrainer(**trainer.to_msg_dict()),
            records=X.identifier,
            targets=y.identifier,
            cv=cv,
            scoring=scoring,
        )
    )

    return FetchableLazyFrame._from_reference(X._client.polars, res)
