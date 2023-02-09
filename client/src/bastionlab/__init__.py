import os
import sys

# Makes generated grpc modules visible
# Needed because internal imports within generated modules are relative
sys.path.append(os.path.join(os.path.dirname(__file__), "pb"))

# Re-exports

from bastionlab.internals.client import Connection, Client
from bastionlab.internals.keys import SigningKey, PublicKey, Identity

from bastionlab.internals.errors import RequestRejected, GRPCException

# Note that we don't reexport bastionlab.polars and bastionlab.torch here
#  as doing so would mean importing polars/torch which is not great
#  when not needed since they take a long time to import (torch in particular)

from .version import __version__ as version

__pdoc__ = {"internals": False, "pb": False}

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
