import os
from setuptools import setup
from setuptools.command.build_py import build_py
import pkg_resources
import re


def read(path):
    return open(os.path.join(os.path.dirname(__file__), path)).read()


DIR = os.path.dirname(__file__) or os.getcwd()
PROTO_FILES = ["bastionlab.proto", "bastionlab_polars.proto", "bastionlab_torch.proto"]
PROTO_PATH = os.path.join(os.path.dirname(DIR), "protos")
LONG_DESCRIPTION = read("README.md")
PKG_NAME = "bastionlab"


def find_version():
    version_file = read(f"src/{PKG_NAME}/version.py")
    version_re = r"__version__ = \"(?P<version>.+)\""
    version = re.match(version_re, version_file).group("version")
    return version


def generate_stub():
    import grpc_tools.protoc

    proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")

    for file in PROTO_FILES:
        print(PROTO_PATH, file)
        res = grpc_tools.protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{proto_include}",
                f"--proto_path={PROTO_PATH}",
                f"--python_out=src/{PKG_NAME}/pb",
                f"--grpc_python_out=src/{PKG_NAME}/pb",
                f"{file}",
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
    name=PKG_NAME,
    version=find_version(),
    description="Client for BastionLab Confidential Analytics.",
    long_description_content_type="text/markdown",
    keywords="confidential computing training client enclave amd-sev machine learning",
    cmdclass={"build_py": BuildPackage},
    long_description=LONG_DESCRIPTION,
    author="Kwabena Amponsem, Lucas Bourtoule",
    author_email="kwabena.amponsem@mithrilsecurity.io, luacs.bourtoule@nithrilsecurity.io",
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=[
        "polars==0.14.24",
        "torch==1.13.1",
        "typing-extensions~=4.4",
        "grpcio==1.47.0",
        "grpcio-tools==1.47.0",
        "colorama~=0.4.6",
        "cryptography~=38.0",
        "seaborn~=0.12.0",
        "pyarrow~=10.0",
        "protobuf==3.20.2",
        "six~=1.16.0",
        "numpy~=1.21",
        "tqdm~=4.64",
    ],
)
