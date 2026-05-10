NVCC = nvcc
CC   = gcc

TARGET = main

SRC_DIR = src
OBJ_DIR = bin

CUDA_SRC = $(SRC_DIR)/main.cu
C_SRC    = $(SRC_DIR)/read_dataset.c

CUDA_OBJ = $(OBJ_DIR)/main.o
C_OBJ    = $(OBJ_DIR)/read_dataset.o

OBJS = $(CUDA_OBJ) $(C_OBJ)

NVCCFLAGS = -O2 -g -G
CFLAGS    = -O2 -g -Wall -Wextra
LDLIBS = -lm

INCLUDES = -I$(SRC_DIR)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(CUDA_OBJ): $(CUDA_SRC) $(SRC_DIR)/read_dataset.h | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(C_OBJ): $(C_SRC) $(SRC_DIR)/read_dataset.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild
