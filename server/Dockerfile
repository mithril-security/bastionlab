FROM ubuntu:18.04

RUN apt update && apt install -y \
    build-essential \
    patchelf \
    libssl-dev \
    pkg-config \
    wget \
    unzip

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.63.0

RUN set -eux; \
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
        amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='5cc9ffd1026e82e7fb2eec2121ad71f4b0f044e88bca39207b3f6b769aaa799c' ;; \
        armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='48c5ecfd1409da93164af20cf4ac2c6f00688b15eb6ba65047f654060c844d85' ;; \
        arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='e189948e396d47254103a49c987e7fb0e5dd8e34b200aa4481ecc4b8e41fb929' ;; \
        i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='0e0be29c560ad958ba52fcf06b3ea04435cb3cd674fbe11ce7d954093b9504fd' ;; \
        *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
    url="https://static.rust-lang.org/rustup/archive/1.25.1/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;

WORKDIR /app

RUN wget 'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip' \
 && unzip 'libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu.zip'

COPY . /app

RUN rustup component add rustfmt \
 && cargo build --release \
 && patchelf --replace-needed libtorch_cpu.so /app/libtorch/lib/libtorch_cpu.so /app/target/release/bastionai_app \
 && patchelf --replace-needed libc10.so /app/libtorch/lib/libc10.so /app/target/release/bastionai_app

RUN mkdir bin/ \
 && cp /app/target/release/bastionai_app bin/ \
 && cp /app/tools/config.toml bin/ \
 && /app/tools/create_tls_config.sh bin/

WORKDIR /app/bin

EXPOSE 50051

CMD ["/app/bin/bastionai_app"]
