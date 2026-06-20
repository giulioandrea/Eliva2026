#ifndef KERNELS_H
#define KERNELS_H

#include "dataset.h"

// Error detection MACRO
#define CHECK_CUDA_ERROR(call)                                                                     \
	{                                                                                              \
		cudaError_t err = call;                                                                    \
		if (err != cudaSuccess) {                                                                  \
			fprintf(stderr, "CUDA error: %s, in file '%s', line%d\n", cudaGetErrorString(err),     \
					__FILE__, __LINE__);                                                           \
			exit(EXIT_FAILURE);                                                                    \
		}                                                                                          \
	}

// kernel launch error check MACRO
#define CHECK_KERNEL_LAUNCH() CHECK_CUDA_ERROR(cudaGetLastError())

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

#define BATCH_SIZE 16

#define OUTPUT_H ((INPUT_H + 2 * PADDING_Y - KERNEL_H) / STRIDE_Y + 1)
#define OUTPUT_W ((INPUT_W + 2 * PADDING_X - KERNEL_W) / STRIDE_X + 1)

#define POOL_SIZE 2
#define POOL_STRIDE 2

#define POOL_OUTPUT_H ((OUTPUT_H - POOL_SIZE) / POOL_STRIDE + 1)
#define POOL_OUTPUT_W ((OUTPUT_W - POOL_SIZE) / POOL_STRIDE + 1)

#define FLATTEN_SIZE (KERNEL_COUNT * POOL_OUTPUT_H * POOL_OUTPUT_W)

#if INPUT_H != INPUT_W || KERNEL_H != KERNEL_W || PADDING_Y != PADDING_X ||                        \
	STRIDE_Y != STRIDE_X || OUTPUT_H != OUTPUT_W || POOL_OUTPUT_H != POOL_OUTPUT_W
#error "The current kernels assume square images, kernels, padding, stride, and outputs."
#endif

#define INPUT_SIZE INPUT_H
#define KERNEL_SIZE KERNEL_H
#define PADDING PADDING_Y
#define STRIDE STRIDE_Y
#define OUTPUT_SIZE OUTPUT_H
#define POOL_OUTPUT_SIZE POOL_OUTPUT_H

#define EPOCHS 10
#define FIXED_SEED 123
#define TILE_SIZE 32

void initializeKernels(float *kernels);

void initializeFullyConnected(float *weights, float *bias);

__global__ void convolutionSharedKernel(float *input, float *kernels, float *output, int batchSize,
										int inputChannels, int inputSize, int kernelSize,
										int kernelCount, int outputSize, int padding, int stride);

__global__ void reluActivationKernel(float *data, int size);

__global__ void maxPoolingKernel(float *input, float *output, int batchSize, int channels,
								 int inputSize, int poolSize, int outputSize, int stride);

__global__ void matrixMultiplySharedKernel(const float *A, const float *B, float *C, int A_rows,
										   int A_cols, int B_cols);

__global__ void addBiasKernel(float *output, float *bias, int rows, int cols);

__global__ void softmaxKernel(float *input, float *output, int batch_size, int num_classes);

__global__ void getPredictionsKernel(float *softmax_output, int *predictions, int batch_size,
									 int num_classes);

__global__ void softmaxCrossEntropyKernel(const float *softmax_output, const int *labels,
										  float *d_logits, int batchSize, int numClasses);

__global__ void fcWeightGradientKernel(const float *input, const float *d_logits, float *d_weights,
									   int batchSize, int inFeatures, int outClasses);

__global__ void fcBiasGradientKernel(const float *d_logits, float *d_bias, int batchSize,
									 int outClasses);

__global__ void fcInputGradientKernel(const float *d_logits, const float *weights, float *d_input,
									  int batchSize, int inFeatures, int outClasses);

__global__ void maxPoolingBackwardKernel(const float *pool_input, const float *d_pool_output,
										 float *d_pool_input, int batchSize, int channels,
										 int inputSize, int poolSize, int outputSize, int stride);

__global__ void reluBackwardKernel(const float *pre_relu, const float *d_output, float *d_input,
								   int size);

__global__ void conv2WeightGradientReduceKernel(const float *input, const float *d_conv_output,
												float *d_kernels, int batchSize, int inputChannels,
												int inputH, int inputW, int kernelCount,
												int kernelH, int kernelW, int outputH, int outputW,
												int paddingY, int paddingX, int strideY,
												int strideX);

__global__ void ridgeL2GradientKernel(float *dW, float *W, int size, float lambda);

__global__ void sgdUpdateKernel(float *params, const float *grads, float learningRate, int size);

float calculateAccuracy(int *predictions, int *labels, int batch_size);

float compute_batch_loss(float *d_softmax_output, int *d_labels, float *d_loss, float *h_loss,
						 int batchSize);

#endif // !KERNELS_H
