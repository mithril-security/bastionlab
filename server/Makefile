SERVER_DIR:=./
BIN_DIR := $(SERVER_DIR)/bin

ifeq ($(MODE),)
	BUILD_TYPE :=
	TARGET_DIR := $(SERVER_DIR)/target/debug
else
	BUILD_TYPE :=--release
	TARGET_DIR := $(SERVER_DIR)/target/release
endif

SHELL := /bin/bash
all: compile

compile:
	@mkdir -p $(BIN_DIR)
	@cd $(SERVER_DIR) && cargo build $(BUILD_TYPE) && cp $(TARGET_DIR)/server $(BIN_DIR)
	
run:
	@./$(BIN_DIR)/server

clean:
	@rm -rf $(BIN_DIR)
fclean:
	make clean
	cargo clean

.PHONY: compile run clean fclean