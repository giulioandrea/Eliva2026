NVCC = nvcc
CC   = gcc
CXX  = g++-15

SRC_DIR = src
OBJ_DIR = build
OUT_DIR = bin

# Three executables. Each one links exactly one object that defines main().
TARGET_MAIN  = $(OUT_DIR)/main
TARGET_FCN   = $(OUT_DIR)/main_fcn
TARGET_CHECK = $(OUT_DIR)/check_fcn_equivalence

TARGETS = $(TARGET_MAIN) $(TARGET_FCN) $(TARGET_CHECK)

NVCCFLAGS = -ccbin=$(CXX) -O2 -g -G -MMD -MP
CFLAGS    = -O2 -g -Wall -Wextra -MMD -MP
LDLIBS    = -lm
INCLUDES  = -I$(SRC_DIR)

# 1. Find all source files dynamically.
SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
SRCS_C  := $(wildcard $(SRC_DIR)/*.c)

# 2. Files that contain main(). Never link two of these into the same executable.
MAIN_CU      := $(SRC_DIR)/main.cu
FCN_MAIN_CU  := $(SRC_DIR)/main_fcn.cu
CHECK_CU     := $(SRC_DIR)/check_fcn_equivalence.cu
ENTRY_CU     := $(MAIN_CU) $(FCN_MAIN_CU) $(CHECK_CU)

# 3. FCN support implementation files, for example kernels_fcn.cu and lenet_fcn.cu.
#    main_fcn.cu is excluded because it contains main().
FCN_SUPPORT_CU := $(filter $(SRC_DIR)/%_fcn.cu,$(filter-out $(FCN_MAIN_CU),$(SRCS_CU)))

# 4. Original CUDA implementation files: every .cu except entrypoints and FCN support.
ORIGINAL_CU := $(filter-out $(ENTRY_CU) $(FCN_SUPPORT_CU),$(SRCS_CU))

# 5. Map source files to object files.
ORIGINAL_CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(ORIGINAL_CU))
FCN_SUPPORT_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(FCN_SUPPORT_CU))
C_OBJS            := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS_C))

OBJ_MAIN      := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(MAIN_CU))
OBJ_FCN_MAIN  := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(FCN_MAIN_CU))
OBJ_CHECK     := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CHECK_CU))

ORIGINAL_OBJS := $(ORIGINAL_CU_OBJS) $(C_OBJS)
FCN_OBJS      := $(ORIGINAL_OBJS) $(FCN_SUPPORT_OBJS)

OBJS_ALL := $(ORIGINAL_OBJS) $(FCN_SUPPORT_OBJS) $(OBJ_MAIN) $(OBJ_FCN_MAIN) $(OBJ_CHECK)
DEPS     := $(OBJS_ALL:.o=.d)

.PHONY: all original fcn check clean rebuild print-objs

all: $(TARGETS)

original: $(TARGET_MAIN)
fcn: $(TARGET_FCN)
check: $(TARGET_CHECK)

# Original executable: original main + original implementation only.
$(TARGET_MAIN): $(OBJ_MAIN) $(ORIGINAL_OBJS) | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

# Fully convolutional executable: FCN main + original implementation + FCN additions.
$(TARGET_FCN): $(OBJ_FCN_MAIN) $(FCN_OBJS) | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

# Equivalence executable: check main + original implementation + FCN additions.
$(TARGET_CHECK): $(OBJ_CHECK) $(FCN_OBJS) | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

# Pattern rule for CUDA files.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Pattern rule for C files.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Create directories if they don't exist.
$(OBJ_DIR) $(OUT_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(OUT_DIR)

rebuild: clean all

# Debug helper: shows which objects each executable will link.
print-objs:
	@echo "ORIGINAL_CU      = $(ORIGINAL_CU)"
	@echo "FCN_SUPPORT_CU   = $(FCN_SUPPORT_CU)"
	@echo "ORIGINAL_OBJS    = $(ORIGINAL_OBJS)"
	@echo "FCN_OBJS         = $(FCN_OBJS)"

# Include automatically generated dependency files.
-include $(DEPS)
