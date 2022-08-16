#######################################
### Base stage: common dependencies ###
#######################################

### base: This image is kept minimal and optimized for size. It has the common runtime dependencies
FROM ubuntu:18.04 AS base

ARG CODENAME=bionic
ARG UBUNTU_VERSION=18.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

# Left here for base setup for AMD SEV (all the different iterations)

### base-build: This image has the common build-time dependencies
FROM base AS base-build

ENV RUST_TOOLCHAIN=stable-x86_64-unknown-linux-gnu

RUN apt update && apt install -y \
    unzip \
    lsb-release \
    debhelper \
    reprepro \
    autoconf \
    automake \
    bison \
    build-essential \
    curl \
    dpkg-dev \
    expect \
    flex \
    gdb \
    git \
    git-core \
    gnupg \
    kmod \
    libboost-system-dev \
    libboost-thread-dev \
    libcurl4-openssl-dev \
    libiptcdata0-dev \
    libjsoncpp-dev \
    liblog4cpp5-dev \
    libprotobuf-dev \
    libssl-dev \
    libtool \
    libxml2-dev \
    ocaml \
    ocamlbuild \
    pkg-config \
    protobuf-compiler \
    python \
    texinfo \
    uuid-dev \
    patchelf\
    wget \
    zip \
    software-properties-common \
    cracklib-runtime \
    && rm -rf /var/lib/apt/lists/*

# -- LIBTORCH
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip &&\
    unzip libtorch-*cpu.zip && rm libtorch-*cpu.zip

# -- Rust
RUN cd /root && \
    curl 'https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init' --output /root/rustup-init && \
    chmod +x /root/rustup-init && \
    echo '1' | /root/rustup-init --default-toolchain $RUST_TOOLCHAIN && \
    echo 'source /root/.cargo/env' >> /root/.bashrc && \
    /root/.cargo/bin/rustup toolchain install $RUST_TOOLCHAIN && \
    /root/.cargo/bin/rustup component add cargo clippy rust-docs rust-src rust-std rustc rustfmt && \
    /root/.cargo/bin/rustup component add --toolchain $RUST_TOOLCHAIN cargo clippy rust-docs rust-src rust-std rustc rustfmt && \
    rm /root/rustup-init
ENV PATH="/root/.cargo/bin:$PATH"

##################################
###   AMD SEV with CPU mode    ###
##################################

FROM base-build as base-cpu

COPY . ./server
RUN make -C server SERVER_DIR=/root/server init && \
    make -C server LIBTORCH_PATH=/root/libtorch MODE=release SERVER_DIR=/root/server &&\
    cp -r ./server/target/release/bastionai_app . &&\
    cp ./server/tools/config.toml . &&\
    cp -r ./server/bin/* .

EXPOSE 50053

CMD ./bastionai_app