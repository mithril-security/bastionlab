from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from bastionai.pb.remote_torch_pb2 import TrainConfig  # type: ignore [import]


@dataclass
class OptimizerConfig:
    lr: float

    def to_msg_dict(self, lr: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class SGD(OptimizerConfig):
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def to_msg_dict(self, lr: Optional[float] = None) -> Dict[str, Any]:
        return {
            "sgd": TrainConfig.SGD(
                learning_rate=lr if lr is not None else self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        }


@dataclass
class Adam(OptimizerConfig):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False

    def to_msg_dict(self, lr: Optional[float] = None) -> Dict[str, Any]:
        return {
            "adam": TrainConfig.Adam(
                learning_rate=lr if lr is not None else self.lr,
                beta_1=self.betas[0],
                beta_2=self.betas[1],
                epsilon=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        }
