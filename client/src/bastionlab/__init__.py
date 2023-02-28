import os
import sys

# Makes generated grpc modules visible
# Needed because internal imports within generated modules are relative
sys.path.append(os.path.join(os.path.dirname(__file__), "pb"))

__pdoc__ = {"pb": False, "version": False}

from .client import Connection, Client
from .keys import SigningKey, PublicKey, Identity
from .errors import RequestRejected, GRPCException

# Note that we don't reexport bastionlab.polars and bastionlab.torch here
#  as doing so would mean importing polars/torch which is not great
#  when not needed since they take a long time to import (torch in particular)

from .version import __version__ as version

__pdoc__["PublicKey.__init__"] = False
__pdoc__["PublicKey.__eq__"] = True
__pdoc__["SigningKey.__init__"] = False
__pdoc__["SigningKey.__eq__"] = True
__pdoc__["Identity.__init__"] = False

__all__ = [
    "Connection",
    "Client",
    "SigningKey",
    "PublicKey",
    "Identity",
    "RequestRejected",
    "GRPCException",
    "version",
]
