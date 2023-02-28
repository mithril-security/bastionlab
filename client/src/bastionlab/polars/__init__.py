__pdoc__ = {}

from .client import BastionLabPolars

__pdoc__["BastionLabPolars.__init__"] = False

from .frame import (
    RemoteLazyFrame,
    RemoteLazyGroupBy,
    FetchableLazyFrame,
    train_test_split,
    Facet,
    RemoteArray,
)

from . import policy
from .frame import train_test_split

__pdoc__["RemoteLazyFrame.__init__"] = False
__pdoc__["RemoteLazyGroupBy.__init__"] = False
__pdoc__["FetchableLazyFrame.__init__"] = False
__pdoc__["Facet.__init__"] = False
__pdoc__["RemoteArray.__init__"] = False

__all__ = [
    "BastionLabPolars",
    "RemoteLazyFrame",
    "RemoteLazyGroupBy",
    "train_test_split",
    "policy",
    "FetchableLazyFrame",
    "train_test_split",
    "Facet",
    "RemoteArray",
]
