# Deploy on premise

The docker images used here are prebuilt ones from our dockerhub, you can take a look at the [build the server from source section](../../quick-tour/quick-tour.ipynb#install-bastionlab-client) if you want to build your own images.


## Running the server

Please make sure you have [Docker](https://docs.docker.com/get-docker/) installed on your machine.

```bash	
docker run \ 
    -p 50056:50056 \
    mithrilsecuritysas/bastionlab:latest
```