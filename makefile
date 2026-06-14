NVCC = nvcc
CC   = gcc

SRC_DIR = src
OBJ_DIR = build
OUT_DIR = bin

CUDA_SRC = $(SRC_DIR)/main.cu
C_SRC    = $(SRC_DIR)/dataset.c

CUDA_OBJ = $(OBJ_DIR)/main.o
C_OBJ    = $(OBJ_DIR)/dataset.o

TARGET = $(OUT_DIR)/main $(OUT_DIR)/dataset

OBJS = $(CUDA_OBJ) $(C_OBJ)

NVCCFLAGS = -ccbin=g++-15 -O2 -g -G
CFLAGS    = -O2 -g -Wall -Wextra
LDLIBS = -lm

INCLUDES = -I$(SRC_DIR)

all: $(TARGET)

$(TARGET): $(OBJS) | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(CUDA_OBJ): $(CUDA_SRC) $(SRC_DIR)/dataset.h | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(C_OBJ): $(C_SRC) $(SRC_DIR)/dataset.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild
