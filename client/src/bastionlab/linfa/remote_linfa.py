from dataclasses import dataclass
from ..pb.bastionlab_linfa_pb2 import ModelResponse
from ..polars.remote_polars import RemoteArray


@dataclass
class FittedModel:
    _identifier: str
    _model_type: str

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def type(self) -> str:
        return self._model_type

    @staticmethod
    def _from_reference(ref: ModelResponse, model_type: str) -> "FittedModel":
        return FittedModel(_identifier=ref.identifier, _model_type=model_type)

    def __repr__(self) -> str:
        return f"FittedModel(identifier={self._identifier})\n  └── {self._model_type}"


@dataclass
class Trainer:
    _fitted_model: "FittedModel" = None
    _name: str = None

    def _to_msg_dict():
        raise NotImplementedError

    def set_name(self, name: str):
        self._name = name

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        client = train_set._client.linfa
        model = client._train(records=train_set, target=target_set, trainer=self)
        self._fitted_model = model

    def predict(self, prediction_input: "RemoteArray"):
        client = prediction_input._client.linfa
        return client._predict(self._fitted_model, prediction_input)

    def predict_proba(self, prediction_input: "RemoteArray"):
        client = prediction_input._client.linfa
        return client._predict_proba(self._fitted_model, prediction_input)

    def __str__(self) -> str:
        return f"{self._fitted_model}"
