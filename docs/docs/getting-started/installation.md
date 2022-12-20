# Installation
____________________________________________

Get started and **install BastionLab Client** and **BastionLab Server**.

## Pre-requisites
___________________________________________

### Technical requirements

To install **BastionLab Client and BastionLab Server**, ensure the following are already installed in your system:

- Python3.7 or greater *(get the latest version of Python at [https://www.python.org/downloads/](https://www.python.org/downloads/) or with your operating system’s package manager)*
- [Python Pip](https://pypi.org/project/pip/) (PyPi), the package manager

To install **BastionLab Server**, you'll also need:

- [Docker](https://www.docker.com/) 

*Here's the [Docker official tutorial](https://docker-curriculum.com/) to set it up on your computer.*

## Installing BastionLab Client
_____________________________________________

### From PyPI

```bash
pip install bastionlab
```

### From source

First, you'll need to clone BastionLab repository:
```bash
git clone https://github.com/mithril-security/bastionlab.git
```
Then install the client library:
```bash
cd ./bastionlab/client
make dev-install
```

## Installing BastionLab Server
______________________________________________

### From PyPI

For **testing purposes only**, BastionLab server can be installed using our pip package.

!!! warning

	This package is meant to quickly setup a running instance of the server and is particularly useful in colab notebooks. It does not provide any mean to configure the server which makes certain features impossible to use (like [authentication](../../../docs/tutorials/authentication/)).

	**For production, please use the Docker image or install the server from source.**
    
```bash
pip install bastionlab-server
```

Once installed, the server can be launched using the following script:

```py
import bastionlab_server
srv = bastionlab_server.start()
```

And stoped this way:

```py
bastionlab_server.stop(srv)
```

### Using the official Docker image

```bash
docker run -p 50056:50056 -d mithrilsecuritysas/bastionlab:latest
```

### By locally building the Docker image

Clone the repository and build the image using the Dockerfile:
```bash
git clone https://github.com/mithril-security/bastionlab.git
cd ./bastionlab/server
docker build -t bastionlab:0.1.0 -t bastionlab:latest .
```
Then run a container based on the image:
```bash
docker run -p 50056:50056 -d bastionlab
```

### From source

First make sure that the following build dependencies (Debian-like systems) are installed on your machine:
```bash
sudo apt update && apt install -y build-essential patchelf libssl-dev pkg-config curl unzip
```

Then, clone our repository:
```bash
git clone https://github.com/mithril-security/bastionlab.git
```
Download and unzip libtorch (Pytorch's C++ backend) from [Pytorch's website](https://pytorch.org/) (you can chose the right build according to your cuda version):
```bash
cd ./bastionlab
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch.zip
```
Lib torch binaries are now available under the libtorch folder. You can now turn to building the server crates:
```bash
cd server
LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make build
```

To run the server, use:
```bash
make run
```
