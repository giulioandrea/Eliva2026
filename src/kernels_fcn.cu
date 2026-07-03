#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels_fcn.h"

__global__ void reshapeFcWeightsToConv2dKernel(const float *fcWeights, float *convWeights,
                                               int inChannels, int kernelH, int kernelW,
                                               int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numClasses * inChannels * kernelH * kernelW;

    if (idx >= total) return;

    int tmp = idx;
    int kx = tmp % kernelW;
    tmp /= kernelW;
    int ky = tmp % kernelH;
    tmp /= kernelH;
    int c = tmp % inChannels;
    tmp /= inChannels;
    int cls = tmp;

    int flattenIndex = (c * kernelH + ky) * kernelW + kx;

    // Conv2D layout: [class, channel, ky, kx]
    // FC layout:     [flattenIndex, class]
    convWeights[idx] = fcWeights[flattenIndex * numClasses + cls];
}

__global__ void fcAsConv2dValidKernel(const float *input, const float *convWeights,
                                      const float *bias, float *logits, int batchSize,
                                      int inChannels, int inputH, int inputW,
                                      int numClasses) {
    extern __shared__ float sharedSum[];

    int cls = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= batchSize || cls >= numClasses) return;

    int features = inChannels * inputH * inputW;
    float localSum = 0.0f;

    for (int f = tid; f < features; f += blockDim.x) {
        int tmp = f;
        int x = tmp % inputW;
        tmp /= inputW;
        int y = tmp % inputH;
        tmp /= inputH;
        int c = tmp;

        int inputIdx = ((b * inChannels + c) * inputH + y) * inputW + x;
        int weightIdx = ((cls * inChannels + c) * inputH + y) * inputW + x;

        localSum += input[inputIdx] * convWeights[weightIdx];
    }

    sharedSum[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedSum[tid] += sharedSum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        logits[b * numClasses + cls] = sharedSum[0] + bias[cls];
    }
}
