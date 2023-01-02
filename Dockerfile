FROM rust:1.65-slim-bullseye AS multistage

RUN rustup component add rustfmt \
    && apt update \
    && apt install -y \
    build-essential \
    patchelf \
    libssl-dev \
    pkg-config \
    wget \
    unzip

WORKDIR /app

RUN wget 'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip' \
    && unzip 'libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip'

COPY ./server /app/server
COPY ./protos /app/protos

ENV LIBTORCH=/app/libtorch
RUN cd server && make

FROM debian:bullseye-slim
COPY --from=multistage /app/server/bin /app/bin
COPY --from=multistage /app/libtorch /app/libtorch

RUN mkdir -p /app/bin/keys

RUN apt update \
    && apt install -y \
    libgomp1

ENV LD_LIBRARY_PATH=/app/libtorch/lib
WORKDIR /app/bin
EXPOSE 50056
CMD ["./bastionlab"]
