from dataclasses import dataclass
from ..pb.bastionlab_linfa_pb2 import ModelResponse


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

    def __str__(self) -> str:
        return f"FittedModel(identifier={self._identifier})\n  └── {self._model_type}"

    def __repr__(self) -> str:
        return str(self)
