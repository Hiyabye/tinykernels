CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -Wpedantic -std=c99
CPPFLAGS ?= -Iinclude
LDLIBS ?= -pthread

OPENMP ?= 0
ifeq ($(OPENMP),1)
  CFLAGS += -fopenmp
  LDFLAGS += -fopenmp
  CPPFLAGS += -DTK_ENABLE_OPENMP=1
else
  CPPFLAGS += -DTK_ENABLE_OPENMP=0
endif

TARGET := tinykernels
SRC_DIR := src
TEST_DIR := tests
BENCH_DIR := results
BUILD_DIR := build
BENCH_DATA_DIR := results/data
BENCH_PLOT_DIR := results/plots
MPLCONFIGDIR ?= $(BUILD_DIR)/matplotlib

SRCS := $(shell find $(SRC_DIR) -name '*.c' | sort)
OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run test bench debug sanitize plots

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./$(TARGET) test

bench: $(TARGET)
	@mkdir -p $(BENCH_DATA_DIR) $(BENCH_PLOT_DIR)
	./$(TARGET) bench
	$(MAKE) plots

plots:
	@mkdir -p $(BENCH_PLOT_DIR) $(MPLCONFIGDIR)
	MPLCONFIGDIR=$(MPLCONFIGDIR) python3 scripts/plot_benchmarks.py $(BENCH_DATA_DIR)/benchmark_results.csv $(BENCH_PLOT_DIR)

debug: CFLAGS := -O0 -g3 -Wall -Wextra -Wpedantic -std=c99
debug: clean $(TARGET)

sanitize: CFLAGS := -O1 -g3 -Wall -Wextra -Wpedantic -std=c99 -fsanitize=address,undefined
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: clean $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)
