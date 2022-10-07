from pathlib import Path
import os
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
import pkg_resources
import re
#from pybind11 import get_cmake_dir,get_include
from pybind11.setup_helpers import Pybind11Extension, build_ext


PROTO_FILES = ["remote_torch.proto","attestation.proto"]
PROTO_PATH = os.path.join(os.path.dirname(__file__), "protos")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

ext_modules = [
    Pybind11Extension("_attestation_c",
        ["attestation_C/lib.cpp"],
        libraries = ['ssl', 'crypto'],
        cxx_std=11)
]

def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

def find_version():
    version_file = read("bastionai/version.py")
    version_re = r"__version__ = \"(?P<version>.+)\""
    version = re.match(version_re, version_file).group("version")
    return version


def generate_stub():
    import grpc_tools.protoc

    proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")
    for file in PROTO_FILES:
        res = grpc_tools.protoc.main(
            [
                "grpc_tools.protoc",
                "-I{}".format(proto_include),
                "--proto_path={}".format(PROTO_PATH),
                "--python_out=bastionai/pb",
                "--grpc_python_out=bastionai/pb",
                "{}".format(file),
            ]
        )
        if res != 0:
            print(f"Proto file generation failed. Cannot continue. Error code: {res}")
            exit(1)


class BuildPackage(build_py):
    def run(self):
        generate_stub()
        super(BuildPackage, self).run()


setup(
    name="bastionai",
    version=find_version(),
    packages=find_packages(),
    description="Client SDK for BastionAI Confidential AI Training.",
    long_description_content_type="text/markdown",
    keywords="confidential computing training client enclave amd-sev machine learning",
    cmdclass={"build_py": BuildPackage},
    long_description=long_description,
    ext_modules=ext_modules,
    author="Kwabena Amponsem, Lucas Bourtoule",
    author_email="kwabena.amponsem@mithrilsecurity.io, luacs.bourtoule@nithrilsecurity.io",
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=[
        "grpcio==1.47.0",
        "grpcio-tools==1.47.0",
        "protobuf==3.20.1",
        "six==1.16.0",
        "torch==1.12.0",
        "numpy==1.23.1",
        "typing-extensions==4.3.0",
        "tqdm==4.64.0",
        "pybind11==2.10.0",
    ],
    package_data={'': ['protos']}
)
