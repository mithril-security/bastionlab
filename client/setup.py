from pathlib import Path
import os
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
import pkg_resources

PROTO_FILES = ["remote_torch.proto"]
PROTO_PATH = os.path.join(os.path.dirname(__file__), "../server/protos")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def generate_stub():
    import grpc_tools.protoc

    if not os.path.exists("bastionai/pb"):
        os.makedirs("bastionai/pb")

    if os.path.exists("bastionai/pb"):
        with open("bastionai/pb/__init__.py", mode="a") as f:
            f.write(
                "import os\nimport sys\n\nsys.path.append(os.path.join(os.path.dirname(__file__)))"
            )

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
    name="bastionai",
    version="0.1.0",
    packages=find_packages(),
    description="Client SDK for BastionAI Confidential AI Training.",
    long_description_content_type="text/markdown",
    keywords="confidential computing training client enclave amd-sev machine learning",
    cmdclass={"build_py": BuildPackage},
    long_description=long_description,
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
    ],
)
