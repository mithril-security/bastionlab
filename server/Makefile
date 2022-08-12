SERVER_DIR:=./
BIN_DIR := $(SERVER_DIR)/bin
SERVER_NAME := server

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

ifeq ($(MODE),)
	BUILD_TYPE :=
	TARGET_DIR := $(SERVER_DIR)/target/debug
else
	BUILD_TYPE :=--release
	TARGET_DIR := $(SERVER_DIR)/target/release
endif


all: compile

compile:
	$(call check_defined, LIBTORCH_PATH)
	@mkdir -p $(BIN_DIR)
	@cd $(SERVER_DIR) && cargo build $(BUILD_TYPE)
	@patchelf --replace-needed libtorch_cpu.so $(LIBTORCH_PATH)/lib/libtorch_cpu.so $(TARGET_DIR)/$(SERVER_NAME)
# Fixme: :(
#	We will have to replace the shared library link for libtorch_gpu too.
	@patchelf --replace-needed libc10.so $(LIBTORCH_PATH)/lib/libc10.so $(TARGET_DIR)/$(SERVER_NAME)
	@cp $(TARGET_DIR)/server $(BIN_DIR)
	
run:
	@./$(BIN_DIR)/server

clean:
	@rm -rf $(BIN_DIR)
fclean:
	make clean
	cargo clean

.PHONY: compile run clean fclean