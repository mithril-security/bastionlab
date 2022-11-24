FROM rust:1.63-slim-bullseye AS multistage

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

# ENV PROTOC=/usr/bin/protoc \
#     PROTOC_INCLUDE=/usr/include

RUN cargo build --release \
 && patchelf --replace-needed libtorch_cpu.so /app/libtorch/lib/libtorch_cpu.so /app/target/release/bastionai_app \
 && patchelf --replace-needed libc10.so /app/libtorch/lib/libc10.so /app/target/release/bastionai_app

RUN mkdir bin/ \
 && cp /app/target/release/bastionai_app bin/ \
 && cp /app/tools/config.toml bin/ \
 && /app/tools/create_tls_config.sh bin/

##

FROM debian:bullseye-slim
COPY --from=multistage /app/bin /app/bin
COPY --from=multistage /app/libtorch /app/libtorch

RUN apt update \
    && apt install -y \
        libgomp1

WORKDIR /app/bin
EXPOSE 50051
CMD ["./bastionai_app"]
