NVCC = nvcc
CC   = gcc

SRC_DIR = src
OBJ_DIR = build
OUT_DIR = bin

TARGET = $(OUT_DIR)/main

NVCCFLAGS = -ccbin=g++-15 -O2 -g -G -MMD -MP
CFLAGS    = -O2 -g -Wall -Wextra -MMD -MP
LDLIBS    = -lm
INCLUDES  = -I$(SRC_DIR)

# 1. Find all source files dynamically
SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
SRCS_C  := $(wildcard $(SRC_DIR)/*.c)

# 2. Map source files to object files in the build directory
OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS_CU)) \
        $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS_C))

# 3. Generate dependency file lists
DEPS := $(OBJS:.o=.d)

all: $(TARGET)

# Link step
$(TARGET): $(OBJS) | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

# Pattern rule for CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Pattern rule for C files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Create directories if they don't exist
$(OBJ_DIR) $(OUT_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(OUT_DIR)

rebuild: clean all

# Include automatically generated dependency files (handles all headers)
-include $(DEPS)

.PHONY: all clean rebuild
