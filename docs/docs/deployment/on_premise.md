# Deploy on premise

## Using our official Docker image

The docker image used here is a prebuilt one from our [dockerhub](https://hub.docker.com/u/mithrilsecuritysas).

This section explains how to work with the docker container provided by Mithril Security.

Launch the server using the docker image :

```bash
docker run \
-p 50051:50051 \
-d mithrilsecuritysas/bastionai:latest # make sure the ports 50051 is available.
```

## By locally building our Docker image

Clone our repository and build the image using our Dockerfile:

```bash	
git clone git@github.com:mithril-security/bastionai.git
cd ./bastionai/server
docker build -t bastionai:0.1.2 -t bastionai:latest .
```

Then simply run a container based on the image:
    
```bash
docker run -it \
    -p 50051:50051 \
    -d bastionai
```

## By locally building from source
First make sure that the following build dependences (Debian-like systems) are installed on your machine:
```bash
sudo apt update && apt install -y build-essential patchelf libssl-dev pkg-config curl unzip
```

Then, clone our repository:
```bash
git clone git@github.com:mithril-security/bastionai.git
```
Download and unzip libtorch (Pytorch's C++ backend) from [Pytorch's website](https://pytorch.org/) (you can chose the right build according to your cuda version):
```bash
cd ./bastionai
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
unzip libtorch.zip
```
Lib torch binaries are now available under the libtorch folder. You can now turn to building the server crates:
```bash
cd server
LIBTORCH_PATH="$(dirname $(pwd))/libtorch" make compile
make copy-bin
make init
```

To run the server, simply use:
```bash
make run
```