CC ?= clang

ifneq ($(LLVM_PREFIX),)
CC := $(LLVM_PREFIX)/bin/clang
CPPFLAGS += -I$(LLVM_PREFIX)/include
LDFLAGS += -L$(LLVM_PREFIX)/lib -Wl,-rpath,$(LLVM_PREFIX)/lib
endif

BASE_CFLAGS := -Wall -Wextra -Wpedantic -std=c11
OPT_CFLAGS := -O3 -march=native
DBG_CFLAGS := -O0 -g3
SAN_CFLAGS := -O1 -g3 -fsanitize=address,undefined

CPPFLAGS += -Iinclude
CFLAGS ?= $(OPT_CFLAGS) $(BASE_CFLAGS)
LDFLAGS += -fopenmp
LDLIBS += -pthread

TARGET := tinykernels
SRC_DIR := src
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run debug sanitize bench

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fopenmp -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

bench: $(TARGET)
	./$(TARGET)
	python3 scripts/plot_benchmarks.py benchmark_results.csv assets

debug: CFLAGS := $(DBG_CFLAGS) $(BASE_CFLAGS)
debug: clean $(TARGET)

sanitize: CFLAGS := $(SAN_CFLAGS) $(BASE_CFLAGS)
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: clean $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)
