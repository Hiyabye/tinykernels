CC := gcc
CFLAGS := -O3 -march=native -Wall -Wextra -Wpedantic -std=c11
CPPFLAGS := -Iinclude
LDFLAGS :=
LDLIBS := -pthread

TARGET := matmul

SRC_DIR := src
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run debug sanitize

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

debug: CFLAGS := -O0 -g3 -Wall -Wextra -Wpedantic -std=c11
debug: clean $(TARGET)

sanitize: CFLAGS := -O1 -g3 -Wall -Wextra -Wpedantic -std=c11 -fsanitize=address,undefined
sanitize: LDFLAGS := -fsanitize=address,undefined
sanitize: clean $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)
