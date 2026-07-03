#include "elements.h"
#include "kernels.h"
#include "kernels_fcn.h"
#include "lenet.h"
#include "lenet_fcn.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

LeNetFCN *LeNetFCN_wrap(LeNet *base) {
    if (!base) {
        fprintf(stderr, "LeNetFCN_wrap received a NULL LeNet pointer.\n");
        return NULL;
    }

    LeNetFCN *fcn = (LeNetFCN *)malloc(sizeof(LeNetFCN));
    if (!fcn) {
        fprintf(stderr, "Host malloc failed in LeNetFCN_wrap.\n");
        exit(EXIT_FAILURE);
    }

    fcn->base = base;
    fcn->d_fc_conv_kernels = NULL;

    CHECK_CUDA_ERROR(cudaMalloc(&fcn->d_fc_conv_kernels,
                                (size_t)fcWeightElements * sizeof(float)));

    LeNetFCN_sync_from_fc(fcn);
    return fcn;
}

void LeNetFCN_free(LeNetFCN *fcn) {
    if (!fcn) return;

    cudaFree(fcn->d_fc_conv_kernels);
    fcn->d_fc_conv_kernels = NULL;
    fcn->base = NULL;
    free(fcn);
}

void LeNetFCN_sync_from_fc(LeNetFCN *fcn) {
    if (!fcn || !fcn->base || !fcn->d_fc_conv_kernels) {
        fprintf(stderr, "LeNetFCN_sync_from_fc received an invalid LeNetFCN wrapper.\n");
        exit(EXIT_FAILURE);
    }

    int total = NUM_CLASSES * KERNEL_COUNT * POOL_OUTPUT_H * POOL_OUTPUT_W;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    reshapeFcWeightsToConv2dKernel<<<gridSize, blockSize>>>(
        fcn->base->d_fc_weights, fcn->d_fc_conv_kernels, KERNEL_COUNT, POOL_OUTPUT_H,
        POOL_OUTPUT_W, NUM_CLASSES);
    CHECK_KERNEL_LAUNCH();
}

void LeNetFCN_forward(float *d_input, int *d_labels, LeNetFCN *fcn, float *timing) {
    LeNetFCN_sync_from_fc(fcn);
    LeNetFCN_forward_no_sync(d_input, d_labels, fcn, timing);
}

void LeNetFCN_forward_no_sync(float *d_input, int *d_labels, LeNetFCN *fcn, float *timing) {
    (void)d_labels;

    if (!fcn || !fcn->base) {
        fprintf(stderr, "LeNetFCN_forward_no_sync received an invalid LeNetFCN wrapper.\n");
        exit(EXIT_FAILURE);
    }

    LeNet *cnn = fcn->base;
    const bool collectTiming = timing != NULL;
    cudaEvent_t start, stop;

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    }

    // Original convolution stage: input -> conv_output.
    dim3 convBlockDim(8, 8);
    dim3 convGridDim((OUTPUT_SIZE + convBlockDim.x - 1) / convBlockDim.x,
                     (OUTPUT_SIZE + convBlockDim.y - 1) / convBlockDim.y,
                     BATCH_SIZE * KERNEL_COUNT);

    int tileSize = convBlockDim.x;
    int tileSizeWithPadding = tileSize + KERNEL_SIZE - 1;
    int sharedMemSize = INPUT_CHANNELS * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);

    if (collectTiming) CHECK_CUDA_ERROR(cudaEventRecord(start));

    convolutionSharedKernel<<<convGridDim, convBlockDim, sharedMemSize>>>(
        d_input, cnn->d_kernels, cnn->d_conv_output, BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
        KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE, PADDING, STRIDE);
    CHECK_KERNEL_LAUNCH();

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timing[0], start, stop));
    }

    // Original ReLU stage: conv_output -> activation.
    int convTotalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
    CHECK_CUDA_ERROR(cudaMemcpy(cnn->d_activation, cnn->d_conv_output,
                                (size_t)convTotalElements * sizeof(float),
                                cudaMemcpyDeviceToDevice));

    int blockSize = 256;
    int reluGridSize = (convTotalElements + blockSize - 1) / blockSize;

    if (collectTiming) CHECK_CUDA_ERROR(cudaEventRecord(start));

    reluActivationKernel<<<reluGridSize, blockSize>>>(cnn->d_activation, convTotalElements);
    CHECK_KERNEL_LAUNCH();

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timing[1], start, stop));
    }

    // Original max-pooling stage: activation -> pooling_output.
    dim3 poolBlockDim(8, 8);
    dim3 poolGridDim((POOL_OUTPUT_SIZE + poolBlockDim.x - 1) / poolBlockDim.x,
                     (POOL_OUTPUT_SIZE + poolBlockDim.y - 1) / poolBlockDim.y,
                     BATCH_SIZE * KERNEL_COUNT);

    if (collectTiming) CHECK_CUDA_ERROR(cudaEventRecord(start));

    maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(cnn->d_activation, cnn->d_pooling_output,
                                                    BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
                                                    POOL_SIZE, POOL_OUTPUT_SIZE, POOL_STRIDE);
    CHECK_KERNEL_LAUNCH();

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timing[2], start, stop));
    }

    // Fully-convolutional classification head.
    // This replaces:
    //   Flatten(pooling_output) + FC(FLATTEN_SIZE, NUM_CLASSES) + bias
    // with:
    //   Conv2D(NUM_CLASSES filters, kernel=POOL_OUTPUT_H x POOL_OUTPUT_W) + bias
    // The output is [BATCH_SIZE, NUM_CLASSES, 1, 1], stored as [BATCH_SIZE, NUM_CLASSES].
    if (collectTiming) CHECK_CUDA_ERROR(cudaEventRecord(start));

    int fcConvBlockSize = 256;
    dim3 fcConvGridDim(NUM_CLASSES, BATCH_SIZE);
    fcAsConv2dValidKernel<<<fcConvGridDim, fcConvBlockSize,
                            fcConvBlockSize * sizeof(float)>>>(
        cnn->d_pooling_output, fcn->d_fc_conv_kernels, cnn->d_fc_bias, cnn->d_logits,
        BATCH_SIZE, KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W, NUM_CLASSES);
    CHECK_KERNEL_LAUNCH();

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timing[3], start, stop));
    }

    // Original softmax and prediction stage.
    int predBlockSize = 256;
    int predGridSize = (BATCH_SIZE + predBlockSize - 1) / predBlockSize;

    if (collectTiming) CHECK_CUDA_ERROR(cudaEventRecord(start));

    softmaxKernel<<<predGridSize, predBlockSize>>>(cnn->d_logits, cnn->d_softmax_output,
                                                   BATCH_SIZE, NUM_CLASSES);
    CHECK_KERNEL_LAUNCH();

    getPredictionsKernel<<<predGridSize, predBlockSize>>>(cnn->d_softmax_output,
                                                          cnn->d_predictions, BATCH_SIZE,
                                                          NUM_CLASSES);
    CHECK_KERNEL_LAUNCH();

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timing[4], start, stop));
    }

    if (collectTiming) {
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    }
}
