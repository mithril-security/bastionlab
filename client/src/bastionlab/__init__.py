import os
import sys

# Makes generated grpc modules visible
# Needed because internal imports within generated modules are relative
sys.path.append(os.path.join(os.path.dirname(__file__), "pb"))

__pdoc__ = {"internals": False, "pb": False, "version": False}

# Re-exports

from bastionlab.internals.client import Connection, Client
Connection.__module__ = __name__
Client.__module__ = __name__

from bastionlab.internals.keys import SigningKey, PublicKey, Identity
SigningKey.__module__ = __name__
PublicKey.__module__ = __name__
Identity.__module__ = __name__

__pdoc__["SigningKey.__init__"] = False
__pdoc__["PublicKey.__init__"] = False

from bastionlab.internals.errors import RequestRejected, GRPCException
RequestRejected.__module__ = __name__
GRPCException.__module__ = __name__

__pdoc__["RequestRejected.__init__"] = False
__pdoc__["GRPCException.__init__"] = False

# Note that we don't reexport bastionlab.polars and bastionlab.torch here
#  as doing so would mean importing polars/torch which is not great
#  when not needed since they take a long time to import (torch in particular)

from .version import __version__ as version

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
