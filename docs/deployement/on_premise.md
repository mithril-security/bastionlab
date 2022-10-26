# Deploy on premise

The docker image used here is a prebuilt one from our [dockerhub](https://hub.docker.com/u/mithrilsecuritysas).

## Simulation mode

This section explains how to work with the docker container provided by Mithril Security. This simulates Intel SGX in software and enables you to run this on any hardware you want.

Launch the server using the simulation docker image:

```bash
docker run -it \
    -p 50051:50051 \
    -p 50052:50052 \ 
    mithrilsecuritysas/bastionai:latest # make sure the ports 50051 and 50052 are available.
```

!!! warning
    Please keep in mind that using this image this way is not secure, since it simulates Intel SGX in software. It is lighter than hardware mode, and should not be used in production.