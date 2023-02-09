__pdoc__ = {}

from bastionlab.internals.polars.client import BastionLabPolars
from bastionlab.internals.polars.remote_polars import (
    RemoteLazyFrame,
    RemoteLazyGroupBy,
    train_test_split,
    FetchableLazyFrame,
    Facet,
    RemoteArray,
)

BastionLabPolars.__module__ = __name__
RemoteLazyFrame.__module__ = __name__
RemoteLazyGroupBy.__module__ = __name__
train_test_split.__module__ = __name__
FetchableLazyFrame.__module__ = __name__
Facet.__module__ = __name__
RemoteArray.__module__ = __name__

__pdoc__["BastionLabPolars.__init__"] = False
__pdoc__["RemoteLazyFrame.__init__"] = False
__pdoc__["RemoteLazyGroupBy.__init__"] = False
__pdoc__["FetchableLazyFrame.__init__"] = False
__pdoc__["Facet.__init__"] = False
__pdoc__["RemoteArray.__init__"] = False


from . import policy
from . import utils

__all__ = [
    "BastionLabPolars",
    "RemoteLazyFrame",
    "RemoteLazyGroupBy",
    "FetchableLazyFrame",
    "train_test_split",
    "Facet",
    "RemoteArray",
    "policy",
    "utils",
]
