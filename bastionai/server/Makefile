SERVER_DIR := $(shell pwd)
BIN_DIR := $(SERVER_DIR)/bin
SERVER_NAME := bastionai_app
TOOLS_DIR := $(SERVER_DIR)/tools
BUILD_TYPE :=
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

ifeq ($(MODE), debug)
	TARGET_DIR := $(SERVER_DIR)/target/debug
endif

ifeq ($(MODE),)
endif

ifeq ($(MODE), release)
	BUILD_TYPE :=--release
	TARGET_DIR := $(SERVER_DIR)/target/release
else
	TARGET_DIR := $(SERVER_DIR)/target/debug
endif


all: init compile copy-bin

init:
# Create server tls config
	@echo Server Directory: $(SERVER_DIR)
	@bash $(TOOLS_DIR)/create_tls_config.sh $(BIN_DIR)

compile:
	$(call check_defined, LIBTORCH_PATH)

	@mkdir -p $(BIN_DIR)
	@cd $(SERVER_DIR) && cargo build $(BUILD_TYPE)
	
# Replace missing LIBTORCH shared library dependency with actual paths
	@patchelf --replace-needed libtorch_cpu.so $(LIBTORCH_PATH)/lib/libtorch_cpu.so $(TARGET_DIR)/$(SERVER_NAME)
	@patchelf --replace-needed libc10.so $(LIBTORCH_PATH)/lib/libc10.so $(TARGET_DIR)/$(SERVER_NAME)

copy-bin:
	@cp $(TOOLS_DIR)/config.toml $(BIN_DIR)
	@cp $(TARGET_DIR)/$(SERVER_NAME) $(BIN_DIR)
	
run:
	@cd $(BIN_DIR) && ./$(SERVER_NAME)

clean:
	@rm -rf $(BIN_DIR)
	
fclean:
	make clean
	cargo clean

.PHONY: compile run clean fclean
