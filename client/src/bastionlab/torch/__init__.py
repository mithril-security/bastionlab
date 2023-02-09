from bastionlab.internals.torch.client import BastionLabTorch
BastionLabTorch.__module__ = __name__

# TODO: Replace TestConfig and TrainConfig with proper python classes/builders
from bastionlab.internals.torch.learner import RemoteLearner, TestConfig, TrainConfig
RemoteLearner.__module__ = __name__
TestConfig.__module__ = __name__
TrainConfig.__module__ = __name__

from bastionlab.internals.torch.remote_torch import (
    RemoteDataset,
    RemoteTensor,
)
RemoteDataset.__module__ = __name__
RemoteTensor.__module__ = __name__

from . import optimizer
from . import utils
from . import psg


__all__ = [
    "BastionLabTorch",
    "RemoteLearner",
    "RemoteDataset",
    "TrainConfig",
    "TestConfig",
    "RemoteTensor",
    "optimizer",
    "utils",
    "psg",
]
