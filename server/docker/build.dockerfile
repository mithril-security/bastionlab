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

ENV RUST_TOOLCHAIN=nightly-x86_64-unknown-linux-gnu

RUN apt update && apt install -y \
    unzip \
    lsb-release \
    debhelper \
    autoconf \
    automake \
    bison \
    build-essential \
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
    libiptcdata0-dev \
    libjsoncpp-dev \
    liblog4cpp5-dev \
    libprotobuf-dev \
    libssl-dev \
    libtool \
    pkg-config \
    protobuf-compiler \
    python \
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
    wget -O /root/rustup-init 'https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init'  && \
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
    make -C server LIBTORCH_PATH=/root/libtorch MODE=release SERVER_DIR=/root/server compile &&\
    cp -r ./server/target/release/bastionai_app . &&\
    cp ./server/tools/config.toml . &&\
    cp -r ./server/bin/* .

EXPOSE 50051

CMD ./bastionai_app

### vscode-dev-env: This image is used for developers to work on bastionai with vscode remote containers extension

FROM base-build AS dev-env

# Options for setup script
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV RUST_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
ENV BASTIONAI_DISABLE_TELEMETRY=1

# run VS Code dev container setup script
COPY ./docker/common-dev.sh /tmp/library-scripts/
RUN bash /tmp/library-scripts/common-dev.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "false" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*  /tmp/library-scripts

USER $USERNAME


# install rustup and cargo for vscode user
RUN cd ~ && \
    curl 'https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init' --output ~/rustup-init && \
    chmod +x ~/rustup-init && \
    echo '1' | ~/rustup-init --default-toolchain $RUST_TOOLCHAIN && \
    . ~/.cargo/env && \
    echo 'source ~/.cargo/env' >> ~/.bashrc && \
    rustup toolchain install $RUST_TOOLCHAIN && \
    rustup component add cargo clippy rust-docs rust-src rust-std rustc rustfmt && \
    rustup component add --toolchain $RUST_TOOLCHAIN cargo clippy rust-docs rust-src rust-std rustc rustfmt && \
    cargo install xargo && \
    rm ~/rustup-init

USER root

# install and configure python and pip
RUN \
    add-apt-repository ppa:deadsnakes/ppa  && \
    apt-get update && \
    apt-get install -y python3.9-dev python3.9-distutils libgl1-mesa-glx && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py && rm get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    pip install virtualenv
