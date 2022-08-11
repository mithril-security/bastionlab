import os
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
import pkg_resources

PROTO_FILES = ["remote_torch.proto"]
PROTO_PATH = os.path.join(os.path.dirname(__file__), "protos")


def generate_stub():
    import grpc_tools.protoc

    proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")
    for file in PROTO_FILES:
        grpc_tools.protoc.main(
            [
                "grpc_tools.protoc",
                "-I{}".format(proto_include),
                "--proto_path={}".format(PROTO_PATH),
                "--python_out=bastionai/pb",
                "--grpc_python_out=bastionai/pb",
                "{}".format(file),
            ]
        )


class BuildPackage(build_py):
    def run(self):
        generate_stub()
        super(BuildPackage, self).run()


setup(
    name='bastionai',
    version="0.1.0",
    packages=find_packages(),
    description="Client SDK for BastionAI Confidential AI Training.",
    long_description_content_type="text/markdown",
    keywords="confidential computing training client enclave amd-sev machine learning",
    cmdclass={"build_py": BuildPackage},
    python_requires=">=3.6.8",
    install_requires=[
        "grpcio==1.47",
        "grpcio-tools==1.47",
        "typing-extensions==4.3.0",
    ],
    extras_require={
        "dev": [
            "setuptools",
            "wheel",
            "check-wheel-contents",
            "auditwheel",
            "grpcio-tools==1.47",
            "grpcio==1.47",
        ]
    },
    classifiers=["Programming Language :: Python :: 3"]
)
