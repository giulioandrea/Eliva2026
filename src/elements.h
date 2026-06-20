#ifndef ELEMENTS_H
#define ELEMENTS_H

#include "dataset.h"
#include "kernels.h"

const int inputElements = BATCH_SIZE * INPUT_CHANNELS * INPUT_H * INPUT_W;

const int convElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_H * OUTPUT_W;

const int poolElements = BATCH_SIZE * KERNEL_COUNT * POOL_OUTPUT_H * POOL_OUTPUT_W;

const int logitsElements = BATCH_SIZE * NUM_CLASSES;

const int kernelElements = KERNEL_COUNT * INPUT_CHANNELS * KERNEL_H * KERNEL_W;

const int fcWeightElements = FLATTEN_SIZE * NUM_CLASSES;

#endif // !ELEMENTS_H
