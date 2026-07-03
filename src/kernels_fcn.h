#ifndef KERNELS_FCN_H
#define KERNELS_FCN_H

#include "dataset.h"
#include "kernels.h"

/*
 * Additive-only FC -> Conv2D support.
 *
 * The original FC weights are laid out as:
 *   fc_weights[flatten_index * NUM_CLASSES + class]
 * where flatten_index = ((channel * POOL_OUTPUT_H) + y) * POOL_OUTPUT_W + x.
 *
 * The equivalent Conv2D weights are laid out as:
 *   conv_weights[((class * KERNEL_COUNT + channel) * POOL_OUTPUT_H + y) * POOL_OUTPUT_W + x]
 * i.e. [NUM_CLASSES, KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W].
 */
__global__ void reshapeFcWeightsToConv2dKernel(const float *fcWeights, float *convWeights,
                                               int inChannels, int kernelH, int kernelW,
                                               int numClasses);

/*
 * Valid Conv2D classification head.
 * Input:  [BATCH_SIZE, KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W]
 * Weight: [NUM_CLASSES, KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W]
 * Output: [BATCH_SIZE, NUM_CLASSES]
 *
 * Because kernelH/kernelW equal the pooled feature map size, the convolution output
 * is 1x1 per class, exactly matching the original FC logits.
 */
__global__ void fcAsConv2dValidKernel(const float *input, const float *convWeights,
                                      const float *bias, float *logits, int batchSize,
                                      int inChannels, int inputH, int inputW,
                                      int numClasses);

#endif // KERNELS_FCN_H
