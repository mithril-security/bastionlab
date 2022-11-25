import os
import sys

# Makes generated grpc modules visible
# Needed because internal imports within generated modules are relative
sys.path.append(os.path.join(os.path.dirname(__file__), "pb"))

from .client import Connection
from .remote_polars import RemoteLazyFrame, RemoteLazyGroupBy
from .keys import SigningKey, PublicKey
