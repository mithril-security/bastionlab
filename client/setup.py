import os
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
import pkg_resources

PROTO_FILES = ["remote_torch.proto"]
PROTO_PATH = os.path.join(os.path.dirname(__file__), "proto")


def generate_stub():
    import grpc_tools.protoc

    proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")
    for file in PROTO_FILES:
        grpc_tools.protoc.main(
            [
                "grpc_tools.protoc",
                "-I{}".format(proto_include),
                "--proto_path={}".format(PROTO_PATH),
                "--python_out=blindai/pb",
                "--grpc_python_out=blindai/pb",
                "{}".format(file),
            ]
        )


class BuildPackage(build_py):
    def run(self):
        generate_stub()
        super(BuildPackage, self).run()


setup(
    name='bastionai',
    packages=find_packages(),
    description="Client SDK for BastionAI Confidential AI Training.",
    classifiers=["Programming Language :: Python :: 3"]
)
