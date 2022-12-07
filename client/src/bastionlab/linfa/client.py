import grpc
from ..pb.bastionlab_linfa_pb2_grpc import LinfaServiceStub
from ..pb.bastionlab_linfa_pb2 import TrainingRequest, PredictionRequest
from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from ..polars import FetchableLazyFrame, BastionLabPolars
    from .trainers import Trainer
    from .remote_linfa import FittedModel


class BastionLabLinfa:
    def __init__(
        self,
        channel: grpc.Channel,
        polars: "BastionLabPolars",
    ) -> None:
        self.stub = LinfaServiceStub(channel)
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

    def predict(self, model: "FittedModel", data: List[float]) -> "FetchableLazyFrame":
        from ..polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(model=model.identifier, data=data, probability=False)
        )
        return FetchableLazyFrame._from_reference(self.polars, res)

    def predict_proba(
        self, model: "FittedModel", data: List[float]
    ) -> "FetchableLazyFrame":
        from ..polars import FetchableLazyFrame

        res = self.stub.Predict(
            PredictionRequest(model=model.identifier, data=data, probability=True)
        )
        return FetchableLazyFrame._from_reference(self.polars, res)
