SERVER_DIR:=./
BIN_DIR := $(SERVER_DIR)/bin
LIBTORCH_PATH := $(LIBTORCH)

ifeq ($(MODE),)
	BUILD_TYPE :=
	TARGET_DIR := $(SERVER_DIR)/target/debug
else
	BUILD_TYPE :=--release
	TARGET_DIR := $(SERVER_DIR)/target/release
endif

all: compile

compile:
	@mkdir -p $(BIN_DIR)
	@export LIBTORCH=$(LIBTORCH_PATH)
	@export LD_LIBRARY_PATH=${LIBTORCH}/lib:${LD_LIBRARY_PATH}
	@cd $(SERVER_DIR) && cargo build $(BUILD_TYPE) && cp $(TARGET_DIR)/server $(BIN_DIR)
	
run:
	@./$(BIN_DIR)/server

clean:
	@rm -rf $(BIN_DIR)
fclean:
	make clean
	cargo clean

.PHONY: compile run clean fclean