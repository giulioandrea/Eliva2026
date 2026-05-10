#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/random.h>
#include "dataset.h"

#define INPUT_H IMAGE_HEIGHT
#define INPUT_W IMAGE_WIDTH
#define INPUT_CHANNELS IMAGE_CHANNELS

#define KERNEL_H 5
#define KERNEL_W 5
#define KERNEL_COUNT 16

#define PADDING_Y 2
#define PADDING_X 2
#define STRIDE_Y 1
#define STRIDE_X 1

#define BATCH_SIZE 32

#define OUTPUT_H ((INPUT_H + 2 * PADDING_Y - KERNEL_H) / STRIDE_Y + 1)
#define OUTPUT_W ((INPUT_W + 2 * PADDING_X - KERNEL_W) / STRIDE_X + 1)

#define POOL_SIZE 2
#define POOL_STRIDE 2

#define POOL_OUTPUT_H ((OUTPUT_H - POOL_SIZE) / POOL_STRIDE + 1)
#define POOL_OUTPUT_W ((OUTPUT_W - POOL_SIZE) / POOL_STRIDE + 1)

#define FLATTEN_SIZE (KERNEL_COUNT * POOL_OUTPUT_H * POOL_OUTPUT_W)

#if INPUT_H != INPUT_W || KERNEL_H != KERNEL_W || PADDING_Y != PADDING_X || \
    STRIDE_Y != STRIDE_X || OUTPUT_H != OUTPUT_W || POOL_OUTPUT_H != POOL_OUTPUT_W
#error "The current kernels assume square images, kernels, padding, stride, and outputs."
#endif

#define INPUT_SIZE INPUT_H
#define KERNEL_SIZE KERNEL_H
#define PADDING PADDING_Y
#define STRIDE STRIDE_Y
#define OUTPUT_SIZE OUTPUT_H
#define POOL_OUTPUT_SIZE POOL_OUTPUT_H

#define NUM_CLASSES 12
#define EPOCHS 10
#define FIXED_SEED 123

// Error detection MACRO
#define CHECK_CUDA_ERROR(call) \
{\
    cudaError_t err = call; \
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA error: %s, in file '%s', line%d\n",\
                cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}

// kernel launch error check MACRO
#define CHECK_KERNEL_LAUNCH() CHECK_CUDA_ERROR(cudaGetLastError())

// Xavier/Glorot initialization
void initializeKernels(float *kernels)
{
    float scale = sqrtf(6.0f / (KERNEL_SIZE * KERNEL_SIZE * (INPUT_CHANNELS + KERNEL_COUNT)));

    for (int k = 0; k < KERNEL_COUNT; k++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    int idx = k * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                              c * KERNEL_SIZE * KERNEL_SIZE +
                              i * KERNEL_SIZE + j;
                    kernels[idx] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;
                }
            }
        }
    }
}

void initializeFullyConnected(float *weights, float *bias)
{
    float scale = sqrtf(6.0f / (FLATTEN_SIZE + NUM_CLASSES));

    for (int i = 0; i < FLATTEN_SIZE * NUM_CLASSES; i++)
        weights[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;
    
    for (int i = 0; i < NUM_CLASSES; i++) bias[i] = 0.0f;
}

// Kernel using shared memory
__global__ void convolutionSharedKernel(
    float * input, float *kernels, float *output,
    int batchSize, int inputChannels, int inputSize,
    int kernelSize, int kernelCount, int outputSize,
    int padding, int stride
)
{
    extern __shared__ float sharedData[];
    // Assuming block.Dim.x == blockDim.y
    int tileSize = blockDim.x;
    int tileSizeWithPadding = tileSize + kernelSize - 1;
    // Output position
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int k = bz % kernelCount;  // Kernel number
    int b = bz / kernelCount; // Batch index

    if (b >= batchSize || k >= kernelCount) return;

    int out_x = bx * tileSize + tx;
    int out_y = by * tileSize + ty;

    int in_x_base = bx * tileSize * stride - padding;
    int in_y_base = by * tileSize * stride - padding;

    float *sharedInput = sharedData;

    // Load data to shared memory
    for (int c = 0; c < inputChannels; c++) {
        for (int dy = 0; dy < tileSizeWithPadding; dy += tileSize) {
            for (int dx = 0; dx < tileSizeWithPadding; dx += tileSize) {
                int in_y = in_y_base + ty + dy;
                int in_x = in_x_base + tx + dx;
                
                float value = 0.0f;
                if (in_y >= 0 && in_y < inputSize && in_x >= 0 && in_x < inputSize)
                    value = input[b * inputChannels * inputSize * inputSize + c * inputSize * inputSize + in_y * inputSize + in_x];

                if (ty + dy < tileSizeWithPadding && tx + dx < tileSizeWithPadding)
                    sharedInput[c * tileSizeWithPadding * tileSizeWithPadding + (ty + dy) * tileSizeWithPadding + (tx + dx)] = value; 
            }
        }
    }
    // Ensure all threads have loaded data to shared memory
    __syncthreads();

    // Convolution
    if (out_x < outputSize && out_y < outputSize && b < batchSize) {
        float sum = 0.0f;

        for (int c = 0; c < inputChannels; c++) {
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int shared_y = ty * stride + ky;
                    int shared_x = tx * stride + kx;

                    float in_val = sharedInput[c * tileSizeWithPadding * tileSizeWithPadding + shared_y * tileSizeWithPadding + shared_x];

                    float kernel_val = kernels[k * inputChannels * kernelSize * kernelSize + c * kernelSize * kernelSize + ky * kernelSize + kx];

                    sum += in_val * kernel_val;
                }
            }
        }

        if (out_x < outputSize && out_y < outputSize)
            output[b * kernelCount * outputSize * outputSize + k * outputSize * outputSize + out_y * outputSize + out_x] = sum;
    }
}

// Simple matrix multiplication without memory tiling
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) sum += A[row * A_cols + k] * B[k * B_cols + col];
        C[row * B_cols + col] = sum;
    }
}

// Progressive bias adjustment
__global__ void addBiasKernel(float *output, float *bias, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) output[row * cols + col] += bias[col];
}

// ReLU function for hidden layers
__global__ void reluActivationKernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void maxPoolingKernel(
    float *input, float *output, int batchSize, int channels,
    int inputSize, int poolSize, int outputSize, int stride
)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize) return;

    int in_x_base = out_x * stride;
    int in_y_base = out_y * stride;

    float maxVal = -FLT_MAX;

    for (int dy = 0; dy < poolSize; dy++) {
        for (int dx = 0; dx < poolSize; dx++) {
            int in_y = in_y_base + dy;
            int in_x = in_x_base + dx;

            if (in_y < inputSize && in_x < inputSize) {
                float value = input[b * channels * inputSize * inputSize + c * inputSize * inputSize + in_y * inputSize + in_x];
                maxVal = fmaxf(maxVal, value);
            }
        }
    }

    output[b * channels * outputSize * outputSize + c * outputSize * outputSize + out_y * outputSize + out_x] = maxVal;
}

// Conversion of raw scores into probabilites using 
// softmax(x_i) = exp(x_i)/sum(x, 0, n-1) for a vector x
__global__ void softmaxKernel(float *input, float *output, int batch_size, int num_classes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);

        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] = expf(input[batch_idx * num_classes + i] - max_val);
            sum += output[batch_idx * num_classes + i];
        }

        for (int i = 0; i < num_classes; i++) output[batch_idx * num_classes + i] /= sum; 
    }
}

// Categorical Cross-Entropy loss for classification
// L = -sum(y_{o,c} * log(p_{o,c})) for each sample o and class c
__global__ void crossEntropyLossKernel(float *predictions, int *labels, float *loss, int batch_size, int num_classes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        int label = labels[batch_idx];
        float pred = predictions[batch_idx * num_classes + label];
        loss[batch_idx] = -logf(pred + 1e-8f); // Avoid log(0)
    }
}

// Combined kernel for softmax and cross-entropy loss (more numerically stable)
__global__ void softmaxCrossEntropyKernel(
    const float *softmax_output, const int *labels, float * d_logits,
    int batchSize, int numClasses
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * numClasses;

    if (idx >= total) return;

    int b = idx / numClasses;
    int c = idx % numClasses;

    float grad = softmax_output[idx];

    if (c == labels[b]) grad -= 1.0f;

    d_logits[idx] = grad / (float)batchSize;
}

// Get class prediction from softmax
__global__ void getPredictionsKernel(float *softmax_output, int *predictions, int batch_size, int num_classes)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        float max_prob = -1.0f;
        int max_idx = -1;

        for (int i = 0; i < num_classes; i++) {
            float prob = softmax_output[batch_idx * num_classes + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = i;
            }
        }

        predictions[batch_idx] = max_idx;
    }
}

// Classification accuracy
float calculateAccuracy(int *predictions, int *labels, int batch_size) {
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {
        if (predictions[i] == labels[i]) correct ++;
    }
    return (float) correct / batch_size;
}

// Forward pass
void forwardCNN(
    float *d_input, float *d_kernels, float *d_conv_output, 
    float *d_activation, float *d_pooling_output, float *timing
)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockDim(8, 8);
    dim3 gridDim(
        (OUTPUT_SIZE + blockDim.x - 1) / blockDim.x,
        (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y,
        BATCH_SIZE * KERNEL_COUNT
    );

    cudaEventRecord(start);

    int tileSize = blockDim.x;
    int tileSizeWithPadding = tileSize + KERNEL_SIZE - 1;
    int sharedMemSize = INPUT_CHANNELS * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);

    convolutionSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
        d_input, d_kernels, d_conv_output,
        BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
        KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
        PADDING, STRIDE
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[0], start, stop);

    int totalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    cudaMemcpy(d_activation, d_conv_output, totalElements * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(start);

    reluActivationKernel<<<gridSize, blockSize>>>(d_activation, totalElements);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[1], start, stop);

    int poolSize = 2;
    int poolStride = 2;
    int poolOutputSize = OUTPUT_SIZE / poolStride;

    dim3 poolBlockDim(8, 8);
    dim3 poolGridDim(
        (poolOutputSize + poolBlockDim.x - 1) / poolBlockDim.x,
        (poolOutputSize + poolBlockDim.y - 1) / poolBlockDim.y,
        BATCH_SIZE * KERNEL_COUNT
    );

    cudaEventRecord(start);

    maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(
        d_activation, d_pooling_output,
        BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
        poolSize, poolOutputSize, poolStride
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[2], start, stop);
}

void forwardCNNClassifier(
    float *d_input,
    float *d_kernels,
    float *d_conv_output,
    float *d_activation,
    float *d_pooling_output,
    float *d_fc_weights,
    float *d_fc_bias,
    float *d_logits,
    float *d_softmax_output,
    int *d_predictions,
    float *timing
)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Convolution
    dim3 convBlockDim(8,8);
    dim3 convGridDim(
        (OUTPUT_SIZE + convBlockDim.x - 1) / convBlockDim.x,
        (OUTPUT_SIZE + convBlockDim.y - 1) / convBlockDim.y,
        BATCH_SIZE * KERNEL_COUNT
    );

    int tileSize = convBlockDim.x;
    int tileSizeWithPadding = tileSize + KERNEL_SIZE - 1;
    int sharedMemSize = INPUT_CHANNELS * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);
    cudaEventRecord(start);
    
    convolutionSharedKernel<<<convGridDim, convBlockDim, sharedMemSize>>>(
        d_input, d_kernels, d_conv_output, BATCH_SIZE, INPUT_CHANNELS,
        INPUT_SIZE, KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
        PADDING, STRIDE
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[0], start, stop);
    
    // Relu
    int convTotalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
    cudaMemcpy(d_activation, d_conv_output, convTotalElements * sizeof(float), cudaMemcpyDeviceToDevice);

    int blockSize = 256;
    int reluGridSize = (convTotalElements + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    reluActivationKernel<<<reluGridSize, blockSize>>>(
        d_activation, convTotalElements
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[1], start, stop);

    // Max pooling
    dim3 poolBlockDim(8, 8);
    dim3 poolGridDim(
        (POOL_OUTPUT_SIZE + poolBlockDim.x - 1) / poolBlockDim.x,
        (POOL_OUTPUT_SIZE + poolBlockDim.y - 1) / poolBlockDim.y,
        BATCH_SIZE * KERNEL_COUNT
    );

    cudaEventRecord(start);

    maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(
        d_activation, d_pooling_output, BATCH_SIZE, KERNEL_COUNT,
        OUTPUT_SIZE, POOL_SIZE, POOL_OUTPUT_SIZE, POOL_STRIDE
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[2], start, stop);

    // Fully connected classification
    dim3 mmBlockDim(16, 16);
    dim3 mmGridDim(
        (NUM_CLASSES + mmBlockDim.x - 1) / mmBlockDim.x,
        (BATCH_SIZE + mmBlockDim.y - 1) / mmBlockDim.y
    );

    cudaEventRecord(start);

    matrixMultiplyKernel<<<mmGridDim, mmBlockDim>>>(
        d_pooling_output, d_fc_weights, d_logits, BATCH_SIZE,
        FLATTEN_SIZE, NUM_CLASSES
    );

    // Add bias
    dim3 biasBlockDim(16, 16);
    dim3 biasGridDim(
        (NUM_CLASSES + biasBlockDim.x - 1) / biasBlockDim.x,
        (BATCH_SIZE + biasBlockDim.y - 1) / biasBlockDim.y
    );

    addBiasKernel<<<biasGridDim, biasBlockDim>>>(
        d_logits, d_fc_bias, BATCH_SIZE, NUM_CLASSES
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[3], start, stop);

    // Softmax
    int predBlockSize = 256;
    int predGridSize = (BATCH_SIZE + predBlockSize - 1) / predBlockSize;

    cudaEventRecord(start);

    softmaxKernel<<<predGridSize, predBlockSize>>>(
        d_logits, d_softmax_output, BATCH_SIZE, NUM_CLASSES
    );

    // Prediction
    getPredictionsKernel<<<predGridSize, predBlockSize>>>(
        d_softmax_output, d_predictions, BATCH_SIZE, NUM_CLASSES
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing[4], start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Backward propagation of fully connected layer
__global__ void fcWeightGradientKernel(
    const float *input, const float *d_logits, float *d_weights,
    int batchSize, int inFeatures, int outClasses
)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (f >= inFeatures || c >= outClasses) return;

    float sum = 0.0f;

    for (int b = 0; b < batchSize; b++) sum += input[b*inFeatures+f]*d_logits[b*outClasses+c];

    d_weights[f * outClasses + c] = sum;
}

__global__ void fcBiasGradientKernel(
    const float *d_logits, float *d_bias, int batchSize, int outClasses
)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= outClasses) return;

    float sum = 0.0f;

    for (int b = 0; b < batchSize; b++) sum += d_logits[b * outClasses + c];

    d_bias[c] = sum;
}

__global__ void fcInputGradientKernel(
    const float *d_logits, const float *weights, float *d_input,
    int batchSize, int inFeatures, int outClasses
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batchSize || f >= inFeatures) return;

    float sum = 0.0f;

    for (int c = 0; c < outClasses; c++) sum += d_logits[b*outClasses+c] * weights[f*outClasses+c];

    d_input[b*inFeatures+f] = sum;
}

// Max pooling backward propagation
__global__ void maxPoolingBackwardKernel(
    const float *pool_input, const float *d_pool_output, float *d_pool_input,
    int batchSize, int channels, int inputSize, int poolSize,
    int outputSize, int stride
)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int bc = blockIdx.z;
    int c = bc % channels;
    int b = bc / channels;

    if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize) return;

    int in_x_base = out_x * stride;
    int in_y_base = out_y * stride;

    float maxVal = -FLT_MAX;
    int max_x = -1;
    int max_y = -1;

    int base = b * channels * inputSize * inputSize + c * inputSize * inputSize;

    for (int dy = 0; dy < poolSize; dy++) {
        for (int dx = 0; dx < poolSize; dx++) {
            int in_y = in_y_base + dy;
            int in_x = in_x_base + dx;

            if (in_y < inputSize && in_x < inputSize) {
                float value = pool_input[base + in_y * inputSize + in_x];

                if (value > maxVal) {
                    maxVal = value;
                    max_y = in_y;
                    max_x = in_x;
                }
            }
        }
    }

    float grad = d_pool_output[
        b * channels * outputSize * outputSize +
        c * outputSize * outputSize +
        out_y * outputSize +
        out_x
    ];

    if (max_x >= 0 && max_y >= 0) atomicAdd(&d_pool_input[base + max_y * inputSize + max_x], grad);
}

// ReLU backward propagation
__global__ void reluBackwardKernel(
    const float *pre_relu, const float *d_output, float *d_input, int size 
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    d_input[idx] = (pre_relu[idx] > 0.0f) ? d_output[idx] : 0.0f;
}

// Kernel convolution with NCHW layout backwardpropagation
__global__ void conv2WeightGradientReduceKernel(
    const float *input, const float *d_conv_output, float *d_kernels,
    int batchSize, int inputChannels, int inputH, int inputW,
    int kernelCount, int kernelH, int kernelW, int outputH, int outputW,
    int paddingY, int paddingX, int strideY, int strideX
)
{
    extern __shared__ float sharedSum[];

    int weightIdx = blockIdx.x;
    int tid = threadIdx.x;

    int totalWeights = kernelCount * inputChannels * kernelH * kernelW;

    if (weightIdx >= totalWeights) return;

    int tmp = weightIdx;

    int kx = tmp % kernelW;
    tmp /= kernelW;

    int ky = tmp %kernelH;
    tmp /= kernelH;

    int c = tmp % inputChannels;
    tmp /= inputChannels;

    int k = tmp;
    float localSum = 0.0f;

    int totalPositions = batchSize * outputH * outputW;

    for (int pos = tid; pos < totalPositions; pos += blockDim.x) {
        int ox = pos % outputW;
        int tmp2 = pos / outputW;
        int oy = tmp2 % outputH;
        int b = tmp2 / outputH;

        int iy = oy * strideY + ky - paddingY;
        int ix = ox * strideX + kx - paddingX;

        if (iy >= 0 && iy < inputH && ix >= 0 && ix < inputW) {
            float inputValue = input[((b * inputChannels + c) * inputH + iy) *inputW + ix];
            float gradValue = d_conv_output[((b * kernelCount + k) * outputH + oy) * outputW + ox];
            localSum += inputValue * gradValue;
        }
    }
    sharedSum[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) sharedSum[tid] += sharedSum[tid + offset];
        __syncthreads();
    }

    if (tid == 0) d_kernels[weightIdx] = sharedSum[0];
}

// Convolution kernel bias backpropagation
__global__ void conv2dBiasGradientReduceKernel(
    const float *d_conv_output, float *d_conv_bias, int batchSize,
    int kernelCount, int outputH, int outputW
)
{
    extern __shared__ float sharedSum[];

    int k = blockIdx.x;
    int tid = threadIdx.x;

    if (k >= kernelCount) return;

    float localSum = 0.0f;

    int totalPositions = batchSize * outputH * outputW;

    for (int pos = tid; pos < totalPositions; pos += blockDim.x) {
        int ox = pos % outputW;
        int tmp = pos / outputW;

        int oy = tmp % outputH;
        int b = tmp / outputH;

        localSum += d_conv_output[((b * kernelCount + k) * outputH + oy) * outputW + ox];
    }

    sharedSum[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) sharedSum[tid] += sharedSum[tid + offset];
        __syncthreads();
    }

    if (tid == 0) d_conv_bias[k] = sharedSum[0];
}

// Stochastic Gradient Descent (SGD)
__global__ void sgdUpdateKernel(
    float *params, const float *grads, float learningRate, int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) params[idx] -= learningRate * grads[idx];
}

// Batch Training
void trainBatch(
    float *d_input, int *d_labels, float *d_kernels, 
    float *d_conv_output, float *d_activation, float *d_pooling_output,
    float *d_fc_weights, float *d_fc_bias, float *d_logits,
    float *d_softmax_output, int *d_predictions, float *d_d_logits,
    float *d_d_fc_weights, float *d_d_fc_bias, float *d_d_pooling_output,
    float *d_d_activation, float *d_d_conv_output, float *d_d_kernels,
    float learningRate
)
{
    float timing[5] = {0.0f};

    forwardCNNClassifier(
        d_input, d_kernels, d_conv_output, d_activation, d_pooling_output,
        d_fc_weights, d_fc_bias, d_logits, d_softmax_output, d_predictions,
        timing
    );

    int blockSize = 256;

    int logitsSize = BATCH_SIZE * NUM_CLASSES;
    int logitsGrid = (logitsSize + blockSize - 1) / blockSize;

    softmaxCrossEntropyKernel<<<logitsGrid, blockSize>>>(
        d_softmax_output, d_labels, d_d_logits, BATCH_SIZE, NUM_CLASSES
    );
    CHECK_KERNEL_LAUNCH();

    dim3 fcWeightBlock(16, 16);
    dim3 fcWeightGrid(
        (NUM_CLASSES + fcWeightBlock.x - 1) / fcWeightBlock.x,
        (FLATTEN_SIZE + fcWeightBlock.y - 1) / fcWeightBlock.y
    );

    fcWeightGradientKernel<<<fcWeightGrid, fcWeightBlock>>>(
        d_pooling_output, d_d_logits, d_d_fc_weights, BATCH_SIZE,
        FLATTEN_SIZE, NUM_CLASSES
    );
    CHECK_KERNEL_LAUNCH();

    int biasGrid = (NUM_CLASSES + blockSize - 1) / blockSize;

    fcBiasGradientKernel<<<biasGrid, blockSize>>>(
        d_d_logits, d_d_fc_bias, BATCH_SIZE, NUM_CLASSES
    );
    CHECK_KERNEL_LAUNCH();

    dim3 fcInputBlock(16, 16);
    dim3 fcInputGrid(
        (FLATTEN_SIZE + fcInputBlock.x - 1) / fcInputBlock.x,
        (BATCH_SIZE + fcInputBlock.y - 1) / fcInputBlock.y
    );

    fcInputGradientKernel<<<fcInputGrid, fcInputBlock>>>(
        d_d_logits, d_fc_weights, d_d_pooling_output, BATCH_SIZE,
        FLATTEN_SIZE, NUM_CLASSES
    );
    CHECK_KERNEL_LAUNCH();

    // d_activation reset for max pooling backward
    int convTotalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
    CHECK_CUDA_ERROR(cudaMemset(d_d_activation, 0, convTotalElements * sizeof(float)));

    dim3 poolBlockDim(8, 8);
    dim3 poolGridDim(
        (POOL_OUTPUT_SIZE + poolBlockDim.x - 1) / poolBlockDim.x,
        (POOL_OUTPUT_SIZE + poolBlockDim.y - 1) / poolBlockDim.y,
        BATCH_SIZE * KERNEL_COUNT
    );

    maxPoolingBackwardKernel<<<poolGridDim, poolBlockDim>>>(
        d_activation, d_d_pooling_output, d_d_activation,
        BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE, POOL_SIZE,
        POOL_OUTPUT_SIZE, POOL_STRIDE
    );
    CHECK_KERNEL_LAUNCH();

    // ReLU backward
    int reluGrid = (convTotalElements + blockSize - 1) / blockSize;
    reluBackwardKernel<<<reluGrid, blockSize>>>(
        d_activation, d_d_activation, d_d_conv_output, convTotalElements
    );
    CHECK_KERNEL_LAUNCH();

    int convWeight = KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    int convWGrid = convWeight;

    conv2WeightGradientReduceKernel<<<convWGrid, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_d_conv_output, d_d_kernels,
        BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE,
        KERNEL_COUNT, KERNEL_SIZE, KERNEL_SIZE, OUTPUT_SIZE, OUTPUT_SIZE,
        PADDING, PADDING, STRIDE, STRIDE
    );
    CHECK_KERNEL_LAUNCH();

    int fcWeights = FLATTEN_SIZE * NUM_CLASSES;
    int fcUpdateGrid = (fcWeights + blockSize - 1) / blockSize;

    sgdUpdateKernel<<<fcUpdateGrid, blockSize>>>(
        d_fc_weights, d_d_fc_weights, learningRate, fcWeights
    );
    CHECK_KERNEL_LAUNCH();

    int fcBiasGrid = (NUM_CLASSES + blockSize - 1) / blockSize;
    sgdUpdateKernel<<<fcBiasGrid, blockSize>>>(
        d_fc_bias, d_d_fc_bias, learningRate, NUM_CLASSES
    );
    CHECK_KERNEL_LAUNCH();

    int convWeightGrid = (convWeight + blockSize - 1) / blockSize;
    sgdUpdateKernel<<<convWeightGrid, blockSize>>>(
        d_kernels, d_d_kernels, learningRate, convWeight
    );
    CHECK_KERNEL_LAUNCH();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Compute average cross-entropy loss for a batch
static float compute_batch_loss(
    float *d_softmax_output,
    int *d_labels,
    float *d_loss,
    float *h_loss,
    int batchSize
)
{
    int blockSize = 256;
    int gridSize = (batchSize + blockSize - 1) / blockSize;

    crossEntropyLossKernel<<<gridSize, blockSize>>>(
        d_softmax_output,
        d_labels,
        d_loss,
        batchSize,
        NUM_CLASSES
    );
    CHECK_KERNEL_LAUNCH();

    CHECK_CUDA_ERROR(cudaMemcpy(
        h_loss,
        d_loss,
        (size_t)batchSize * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float sum = 0.0f;

    for (int i = 0; i < batchSize; i++) {
        sum += h_loss[i];
    }

    return sum / (float)batchSize;
}

int main(int argc, char **argv)
{
    // Uncomment to use getrandom for better randomness in shuffling
    // unsigned char buffer[16];
    // getrandom(buffer, sizeof(buffer), 0);
    // srand(*(unsigned int *)buffer);
    
    // Comment out to ensure reproducibility for testing and benchmarking
    srand(FIXED_SEED);

    const char *datasetRoot = (argc > 1) ? argv[1] : "dataset";

    char trainPath[4096];
    char validationPath[4096];

    snprintf(trainPath, sizeof(trainPath), "%s/training", datasetRoot);
    snprintf(validationPath, sizeof(validationPath), "%s/validation", datasetRoot);

    Dataset train;
    Dataset validation;

    if (!load_dataset_index(&train, trainPath)) {
        fprintf(stderr, "Failed to load training dataset from '%s'.\n", trainPath);
        return EXIT_FAILURE;
    }

    if (!load_dataset_index(&validation, validationPath)) {
        fprintf(stderr, "Failed to load validation dataset from '%s'.\n", validationPath);
        free_dataset(&train);
        return EXIT_FAILURE;
    }

    printf("\nTRAIN DATASET\n");
    print_dataset_info(&train);

    printf("\nVALIDATION DATASET\n");
    print_dataset_info(&validation);

    int trainBatches = train.count / BATCH_SIZE;
    int validationBatches = validation.count / BATCH_SIZE;

    if (trainBatches <= 0) {
        fprintf(stderr, "Training dataset too small for BATCH_SIZE=%d.\n", BATCH_SIZE);
        free_dataset(&train);
        free_dataset(&validation);
        return EXIT_FAILURE;
    }

    printf("\nNetwork configuration:\n");
    printf("Input:        %d x %d x %d\n", INPUT_CHANNELS, INPUT_H, INPUT_W);
    printf("Conv output:  %d x %d x %d\n", KERNEL_COUNT, OUTPUT_H, OUTPUT_W);
    printf("Pool output:  %d x %d x %d\n", KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W);
    printf("Flatten size: %d\n", FLATTEN_SIZE);
    printf("Classes:      %d\n", NUM_CLASSES);
    printf("Batch size:   %d\n", BATCH_SIZE);
    printf("Epochs:       %d\n\n", EPOCHS);

    const int inputElements =
        BATCH_SIZE * INPUT_CHANNELS * INPUT_H * INPUT_W;

    const int convElements =
        BATCH_SIZE * KERNEL_COUNT * OUTPUT_H * OUTPUT_W;

    const int poolElements =
        BATCH_SIZE * KERNEL_COUNT * POOL_OUTPUT_H * POOL_OUTPUT_W;

    const int logitsElements =
        BATCH_SIZE * NUM_CLASSES;

    const int kernelElements =
        KERNEL_COUNT * INPUT_CHANNELS * KERNEL_H * KERNEL_W;

    const int fcWeightElements =
        FLATTEN_SIZE * NUM_CLASSES;

    float *h_input = (float *)malloc((size_t)inputElements * sizeof(float));
    int *h_labels = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
    int *h_predictions = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
    float *h_loss = (float *)malloc((size_t)BATCH_SIZE * sizeof(float));

    float *h_kernels = (float *)malloc((size_t)kernelElements * sizeof(float));
    float *h_fc_weights = (float *)malloc((size_t)fcWeightElements * sizeof(float));
    float *h_fc_bias = (float *)malloc((size_t)NUM_CLASSES * sizeof(float));

    if (!h_input || !h_labels || !h_predictions || !h_loss ||
        !h_kernels || !h_fc_weights || !h_fc_bias) {
        fprintf(stderr, "Host malloc failed.\n");

        free(h_input);
        free(h_labels);
        free(h_predictions);
        free(h_loss);
        free(h_kernels);
        free(h_fc_weights);
        free(h_fc_bias);

        free_dataset(&train);
        free_dataset(&validation);

        return EXIT_FAILURE;
    }

    initializeKernels(h_kernels);
    initializeFullyConnected(h_fc_weights, h_fc_bias);

    float *d_input = NULL;
    int *d_labels = NULL;

    float *d_kernels = NULL;
    float *d_conv_output = NULL;
    float *d_activation = NULL;
    float *d_pooling_output = NULL;

    float *d_fc_weights = NULL;
    float *d_fc_bias = NULL;
    float *d_logits = NULL;
    float *d_softmax_output = NULL;
    int *d_predictions = NULL;

    float *d_d_logits = NULL;
    float *d_d_fc_weights = NULL;
    float *d_d_fc_bias = NULL;
    float *d_d_pooling_output = NULL;
    float *d_d_activation = NULL;
    float *d_d_conv_output = NULL;
    float *d_d_kernels = NULL;

    float *d_loss = NULL;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, (size_t)inputElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_labels, (size_t)BATCH_SIZE * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_kernels, (size_t)kernelElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_conv_output, (size_t)convElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_activation, (size_t)convElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pooling_output, (size_t)poolElements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_fc_weights, (size_t)fcWeightElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_fc_bias, (size_t)NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_logits, (size_t)logitsElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_softmax_output, (size_t)logitsElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, (size_t)BATCH_SIZE * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_logits, (size_t)logitsElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_fc_weights, (size_t)fcWeightElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_fc_bias, (size_t)NUM_CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_pooling_output, (size_t)poolElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_activation, (size_t)convElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_conv_output, (size_t)convElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_kernels, (size_t)kernelElements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_loss, (size_t)BATCH_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(
        d_kernels,
        h_kernels,
        (size_t)kernelElements * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMemcpy(
        d_fc_weights,
        h_fc_weights,
        (size_t)fcWeightElements * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMemcpy(
        d_fc_bias,
        h_fc_bias,
        (size_t)NUM_CLASSES * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const float learningRate = 0.001f;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle_dataset(&train);

        float trainLossSum = 0.0f;
        float trainAccuracySum = 0.0f;
        int usedTrainBatches = 0;

        for (int batch = 0; batch < trainBatches; batch++) {
            int startIndex = batch * BATCH_SIZE;

            int loaded = load_batch(
                &train,
                startIndex,
                BATCH_SIZE,
                h_input,
                h_labels
            );

            if (loaded != BATCH_SIZE) {
                continue;
            }

            CHECK_CUDA_ERROR(cudaMemcpy(
                d_input,
                h_input,
                (size_t)inputElements * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CHECK_CUDA_ERROR(cudaMemcpy(
                d_labels,
                h_labels,
                (size_t)BATCH_SIZE * sizeof(int),
                cudaMemcpyHostToDevice
            ));

            trainBatch(
                d_input,
                d_labels,
                d_kernels,
                d_conv_output,
                d_activation,
                d_pooling_output,
                d_fc_weights,
                d_fc_bias,
                d_logits,
                d_softmax_output,
                d_predictions,
                d_d_logits,
                d_d_fc_weights,
                d_d_fc_bias,
                d_d_pooling_output,
                d_d_activation,
                d_d_conv_output,
                d_d_kernels,
                learningRate
            );

            float batchLoss = compute_batch_loss(
                d_softmax_output,
                d_labels,
                d_loss,
                h_loss,
                BATCH_SIZE
            );

            CHECK_CUDA_ERROR(cudaMemcpy(
                h_predictions,
                d_predictions,
                (size_t)BATCH_SIZE * sizeof(int),
                cudaMemcpyDeviceToHost
            ));

            float batchAccuracy = calculateAccuracy(
                h_predictions,
                h_labels,
                BATCH_SIZE
            );

            trainLossSum += batchLoss;
            trainAccuracySum += batchAccuracy;
            usedTrainBatches++;

            if ((batch + 1) % 10 == 0) {
                printf(
                    "Epoch %d/%d | Batch %d/%d | Train loss %.4f | Train acc %.4f\n",
                    epoch + 1,
                    EPOCHS,
                    batch + 1,
                    trainBatches,
                    batchLoss,
                    batchAccuracy
                );
            }
        }

        float avgTrainLoss = 0.0f;
        float avgTrainAccuracy = 0.0f;

        if (usedTrainBatches > 0) {
            avgTrainLoss = trainLossSum / (float)usedTrainBatches;
            avgTrainAccuracy = trainAccuracySum / (float)usedTrainBatches;
        }

        float validationLossSum = 0.0f;
        float validationAccuracySum = 0.0f;
        int usedValidationBatches = 0;

        for (int batch = 0; batch < validationBatches; batch++) {
            int startIndex = batch * BATCH_SIZE;

            int loaded = load_batch(
                &validation,
                startIndex,
                BATCH_SIZE,
                h_input,
                h_labels
            );

            if (loaded != BATCH_SIZE) {
                continue;
            }

            CHECK_CUDA_ERROR(cudaMemcpy(
                d_input,
                h_input,
                (size_t)inputElements * sizeof(float),
                cudaMemcpyHostToDevice
            ));

            CHECK_CUDA_ERROR(cudaMemcpy(
                d_labels,
                h_labels,
                (size_t)BATCH_SIZE * sizeof(int),
                cudaMemcpyHostToDevice
            ));

            float timing[5] = {0.0f};

            forwardCNNClassifier(
                d_input,
                d_kernels,
                d_conv_output,
                d_activation,
                d_pooling_output,
                d_fc_weights,
                d_fc_bias,
                d_logits,
                d_softmax_output,
                d_predictions,
                timing
            );

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            float batchLoss = compute_batch_loss(
                d_softmax_output,
                d_labels,
                d_loss,
                h_loss,
                BATCH_SIZE
            );

            CHECK_CUDA_ERROR(cudaMemcpy(
                h_predictions,
                d_predictions,
                (size_t)BATCH_SIZE * sizeof(int),
                cudaMemcpyDeviceToHost
            ));

            float batchAccuracy = calculateAccuracy(
                h_predictions,
                h_labels,
                BATCH_SIZE
            );

            validationLossSum += batchLoss;
            validationAccuracySum += batchAccuracy;
            usedValidationBatches++;
        }

        float avgValidationLoss = 0.0f;
        float avgValidationAccuracy = 0.0f;

        if (usedValidationBatches > 0) {
            avgValidationLoss = validationLossSum / (float)usedValidationBatches;
            avgValidationAccuracy = validationAccuracySum / (float)usedValidationBatches;
        }

        printf(
            "\nEpoch %d/%d completed | "
            "Train loss %.4f | Train acc %.4f | Used train batches %d/%d | "
            "Val loss %.4f | Val acc %.4f | Used val batches %d/%d\n\n",
            epoch + 1,
            EPOCHS,
            avgTrainLoss,
            avgTrainAccuracy,
            usedTrainBatches,
            trainBatches,
            avgValidationLoss,
            avgValidationAccuracy,
            usedValidationBatches,
            validationBatches
        );
    }

    cudaFree(d_input);
    cudaFree(d_labels);

    cudaFree(d_kernels);
    cudaFree(d_conv_output);
    cudaFree(d_activation);
    cudaFree(d_pooling_output);

    cudaFree(d_fc_weights);
    cudaFree(d_fc_bias);
    cudaFree(d_logits);
    cudaFree(d_softmax_output);
    cudaFree(d_predictions);

    cudaFree(d_d_logits);
    cudaFree(d_d_fc_weights);
    cudaFree(d_d_fc_bias);
    cudaFree(d_d_pooling_output);
    cudaFree(d_d_activation);
    cudaFree(d_d_conv_output);
    cudaFree(d_d_kernels);

    cudaFree(d_loss);

    free(h_input);
    free(h_labels);
    free(h_predictions);
    free(h_loss);

    free(h_kernels);
    free(h_fc_weights);
    free(h_fc_bias);

    free_dataset(&train);
    free_dataset(&validation);

    return EXIT_SUCCESS;
}
