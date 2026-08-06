CC       ?= gcc
OPT      ?= -O3 -march=native
WARN      = -Wall -Wextra -Wpedantic -Werror -Wshadow -Wformat=2 -Wstrict-prototypes \
            -Wmissing-prototypes -Wwrite-strings -Wundef -Wpointer-arith
CFLAGS   ?= $(OPT) $(WARN) -std=c99
CPPFLAGS ?= -Iinclude
LDLIBS   ?= -pthread

OPENMP ?= 0
ifeq ($(OPENMP),1)
  CFLAGS += -fopenmp
  LDFLAGS += -fopenmp
  CPPFLAGS += -DENABLE_OPENMP=1
else
  CPPFLAGS += -DENABLE_OPENMP=0
endif

TARGET := tinykernels
SRCS := $(shell find src -name '*.c' | sort)
OBJS := $(patsubst %.c,build/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# matplotlib lives in the project venv when present; fall back to system python3.
PYTHON := $(shell if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

.PHONY: all clean test bench plots debug sanitize format

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf build $(TARGET)

test: $(TARGET)
	./$(TARGET) test

bench: $(TARGET)
	@mkdir -p results/data results/plots build/matplotlib
	./$(TARGET) bench
	$(MAKE) plots

plots:
	@mkdir -p results/plots build/matplotlib
	MPLCONFIGDIR=build/matplotlib $(PYTHON) scripts/plot_benchmarks.py results/data/benchmark_results.csv results/plots

debug: CFLAGS := -O0 -g3 $(WARN) -std=c99
debug: clean $(TARGET)

sanitize: CFLAGS := -O1 -g3 $(WARN) -std=c99 -fsanitize=address,undefined -fno-omit-frame-pointer
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: clean $(TARGET)

format:
	clang-format -i $$(find src include -name '*.c' -o -name '*.h')

-include $(DEPS)
