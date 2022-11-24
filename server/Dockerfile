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

RUN wget 'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip' \
    && unzip 'libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip'

COPY . /app

RUN cd server && make \
    && patchelf --replace-needed libtorch_cpu.so /app/libtorch/lib/libtorch_cpu.so /app/server/target/release/bastionlab \
    && patchelf --replace-needed libc10.so /app/libtorch/lib/libc10.so /app/server/target/release/bastionlab

RUN mkdir -p /app/bin/keys
RUN mkdir -p /app/bin
RUN cp /app/server/target/release/bastionlab /app/bin

FROM debian:bullseye-slim
COPY --from=multistage /app/bin /app/bin
COPY --from=multistage /app/libtorch /app/libtorch

RUN apt update \
    && apt install -y \
    libgomp1

WORKDIR /app/bin
EXPOSE 50056
CMD ["./bastionlab"]
