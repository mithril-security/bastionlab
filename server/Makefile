.PHONY: all build install run clean

ifeq ($(origin LIBTORCH), undefined)
    $(info "LIBTORCH environement variable not set, using default: ../libtorch")
    LIBTORCH := $(shell dirname $(shell pwd))/libtorch
endif

LD_LIBRARY_PATH := $(LIBTORCH)/lib:$(LD_LIBRARY_PATH)
PKG_NAME := bastionlab

ifeq ($(origin CUDA), undefined)
    $(info "CUDA environement variable not set, building against a GPU version of libtorch will likely fail")
else
    LD_LIBRARY_PATH := $(CUDA)/lib64:$(LD_LIBRARY_PATH)
    $(info $(LD_LIBRARY_PATH))
endif

all: build install

bin:
	mkdir bin

bin/config.toml: bin
	cp tools/config.toml bin/

bin/tls/host_server.pem bin/tls/host_server.key &: bin
	mkdir -p bin/tls && openssl req -newkey rsa:2048 -nodes -keyout bin/tls/host_server.key  \
		-x509 -days 365 -out bin/tls/host_server.pem \
		-subj "/C=FR/CN=bastionlab-server"

keys_directory:
	mkdir keys/owners -p
	mkdir keys/users -p

build:
	LIBTORCH=$(LIBTORCH) CUDA=$(CUDA) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) cargo build --release 

target/release/$(PKG_NAME):
	LIBTORCH=$(LIBTORCH) CUDA=$(CUDA) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) cargo build --release

bin/$(PKG_NAME): target/release/$(PKG_NAME)
	cp target/release/$(PKG_NAME) bin/
	
install: bin/tls/host_server.pem bin/tls/host_server.key bin/config.toml keys_directory
	cp target/release/$(PKG_NAME) bin/

run: bin/$(PKG_NAME) bin/tls/host_server.pem bin/tls/host_server.key bin/config.toml
	LIBTORCH=$(LIBTORCH) CUDA=$(CUDA) LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) sh -c 'cd bin && ./$(PKG_NAME)'

clean:
	rm -rf bin/
	cargo clean
