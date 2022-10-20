.PHONY: all build install run clean

ifeq ($(origin LIBTORCH), undefined)
	$(info "LIBTORCH environement variable not set, using default: ../libtorch")
	export LIBTORCH=$(shell dirname $(shell pwd))/libtorch
endif

ifeq ($(origin CUDA), undefined)
	$(info "CUDA environement variable not set, building against a GPU version of libtorch will likely fail")
endif

all: build install

bin:
	mkdir bin

bin/config.toml: bin
	cp tools/config.toml bin/

bin/tls/host_server.pem bin/tls/host_server.key &: bin
	./tools/create_tls_config.sh bin/

build:
	cargo build --release

target/release/bastionai_app:
	cargo build --release

bin/bastionai_app: target/release/bastionai_app
	cp target/release/bastionai_app bin/

install: bin/tls/host_server.pem bin/tls/host_server.key bin/config.toml
	cp target/release/bastionai_app bin/

run: bin/bastionai_app bin/tls/host_server.pem bin/tls/host_server.key bin/config.toml
	cd bin && ./bastionai_app

clean:
	rm -rf bin/
	cargo clean
