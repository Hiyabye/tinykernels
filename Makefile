CC ?= cc

CFLAGS ?= -O3 -march=native -Wall -Wextra -Wpedantic -std=c11
CPPFLAGS ?= -Iinclude
CPPFLAGS += -D_POSIX_C_SOURCE=200809L
LDFLAGS ?=
LDLIBS ?= -pthread

OPENMP ?= 1
ifeq ($(OPENMP),1)
  CFLAGS += -fopenmp
  LDFLAGS += -fopenmp
  CPPFLAGS += -DTK_ENABLE_OPENMP=1
else
  CPPFLAGS += -DTK_ENABLE_OPENMP=0
endif

ifneq ($(LLVM_PREFIX),)
  CC := $(LLVM_PREFIX)/bin/clang
  CPPFLAGS += -I$(LLVM_PREFIX)/include
  LDFLAGS += -L$(LLVM_PREFIX)/lib -Wl,-rpath,$(LLVM_PREFIX)/lib
endif

TARGET := tinykernels
SRC_DIR := src
BUILD_DIR := build

SRCS := $(shell find $(SRC_DIR) -name '*.c' | sort)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run test bench debug sanitize plots

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET)

bench: $(TARGET)
	./$(TARGET)
	$(MAKE) plots

plots:
	python3 scripts/plot_benchmarks.py benchmark_results.csv assets

debug: CFLAGS := -O0 -g3 -Wall -Wextra -Wpedantic -std=c11
debug: clean $(TARGET)

sanitize: CFLAGS := $(SAN_CFLAGS) $(BASE_CFLAGS)
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: clean $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)
