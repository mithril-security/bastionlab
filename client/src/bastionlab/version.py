"""
This module exists only to simplify retrieving the version number of bastionlab and torch
as well as the URL to install torch Pre-cxx11 ABI and cxx11 ABI.
"""

_base_url = "https://download.pytorch.org/libtorch/cpu/"

__all__ = [
    "__version__",
    "__torch_version__",
    "__torch_url__",
    "__torch_cxx11_url__",
]

__version__ = "0.3.7"
__torch_version__ = "1.13.1"
__torch_url__ = (
    _base_url + "libtorch-shared-with-deps-" + __torch_version__ + "%2Bcpu.zip"
)
__torch_cxx11_url__ = (
    _base_url
    + "libtorch-cxx11-abi-shared-with-deps-"
    + __torch_version__
    + "%2Bcpu.zip"
)
