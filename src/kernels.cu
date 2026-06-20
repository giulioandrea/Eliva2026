#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "kernels.h"

// Xavier/Glorot initialization
void initializeKernels(float *kernels) {
	float scale = sqrtf(6.0f / (KERNEL_SIZE * KERNEL_SIZE * (INPUT_CHANNELS + KERNEL_COUNT)));

	for (int k = 0; k < KERNEL_COUNT; k++) {
		for (int c = 0; c < INPUT_CHANNELS; c++) {
			for (int i = 0; i < KERNEL_SIZE; i++) {
				for (int j = 0; j < KERNEL_SIZE; j++) {
					int idx = k * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
							  c * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j;
					kernels[idx] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;
				}
			}
		}
	}
}

void initializeFullyConnected(float *weights, float *bias) {
	float scale = sqrtf(6.0f / (FLATTEN_SIZE + NUM_CLASSES));

	for (int i = 0; i < FLATTEN_SIZE * NUM_CLASSES; i++)
		weights[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * scale;

	for (int i = 0; i < NUM_CLASSES; i++)
		bias[i] = 0.0f;
}

// Kernel using shared memory
__global__ void convolutionSharedKernel(float *input, float *kernels, float *output, int batchSize,
										int inputChannels, int inputSize, int kernelSize,
										int kernelCount, int outputSize, int padding, int stride) {
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
	int k = bz % kernelCount; // Kernel number
	int b = bz / kernelCount; // Batch index

	if (b >= batchSize || k >= kernelCount)
		return;

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
					value = input[b * inputChannels * inputSize * inputSize +
								  c * inputSize * inputSize + in_y * inputSize + in_x];

				if (ty + dy < tileSizeWithPadding && tx + dx < tileSizeWithPadding)
					sharedInput[c * tileSizeWithPadding * tileSizeWithPadding +
								(ty + dy) * tileSizeWithPadding + (tx + dx)] = value;
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

					float in_val = sharedInput[c * tileSizeWithPadding * tileSizeWithPadding +
											   shared_y * tileSizeWithPadding + shared_x];

					float kernel_val = kernels[k * inputChannels * kernelSize * kernelSize +
											   c * kernelSize * kernelSize + ky * kernelSize + kx];

					sum += in_val * kernel_val;
				}
			}
		}

		if (out_x < outputSize && out_y < outputSize)
			output[b * kernelCount * outputSize * outputSize + k * outputSize * outputSize +
				   out_y * outputSize + out_x] = sum;
	}
}

// Matrix multiplication using shared-memory tiling
__global__ void matrixMultiplySharedKernel(const float *A, const float *B, float *C, int A_rows,
										   int A_cols, int B_cols) {
	__shared__ float sA[TILE_SIZE][TILE_SIZE];
	__shared__ float sB[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float sum = 0.0f;
	int numTiles = (A_cols + TILE_SIZE - 1) / TILE_SIZE;

	for (int tile = 0; tile < numTiles; tile++) {
		int aCol = tile * TILE_SIZE + threadIdx.x;
		int bRow = tile * TILE_SIZE + threadIdx.y;

		if (row < A_rows && aCol < A_cols) {
			sA[threadIdx.y][threadIdx.x] = A[row * A_cols + aCol];
		} else {
			sA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (bRow < A_cols && col < B_cols) {
			sB[threadIdx.y][threadIdx.x] = B[bRow * B_cols + col];
		} else {
			sB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; k++) {
			sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
		}

		// Prevent any thread from overwriting the shared tile for the next
		// phase while other threads are still using the current tile.
		__syncthreads();
	}

	if (row < A_rows && col < B_cols) {
		C[row * B_cols + col] = sum;
	}
}

// Progressive bias adjustment
__global__ void addBiasKernel(float *output, float *bias, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows && col < cols)
		output[row * cols + col] += bias[col];
}

// ReLU function for hidden layers
__global__ void reluActivationKernel(float *data, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
		data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void maxPoolingKernel(float *input, float *output, int batchSize, int channels,
								 int inputSize, int poolSize, int outputSize, int stride) {
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.z % channels;
	int b = blockIdx.z / channels;

	if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize)
		return;

	int in_x_base = out_x * stride;
	int in_y_base = out_y * stride;

	float maxVal = -FLT_MAX;

	for (int dy = 0; dy < poolSize; dy++) {
		for (int dx = 0; dx < poolSize; dx++) {
			int in_y = in_y_base + dy;
			int in_x = in_x_base + dx;

			if (in_y < inputSize && in_x < inputSize) {
				float value = input[b * channels * inputSize * inputSize +
									c * inputSize * inputSize + in_y * inputSize + in_x];
				maxVal = fmaxf(maxVal, value);
			}
		}
	}

	output[b * channels * outputSize * outputSize + c * outputSize * outputSize +
		   out_y * outputSize + out_x] = maxVal;
}

// Conversion of raw scores into probabilites using
// softmax(x_i) = exp(x_i)/sum(x, 0, n-1) for a vector x
__global__ void softmaxKernel(float *input, float *output, int batch_size, int num_classes) {
	int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (batch_idx < batch_size) {
		float max_val = -FLT_MAX;
		for (int i = 0; i < num_classes; i++)
			max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);

		float sum = 0.0f;
		for (int i = 0; i < num_classes; i++) {
			output[batch_idx * num_classes + i] =
				expf(input[batch_idx * num_classes + i] - max_val);
			sum += output[batch_idx * num_classes + i];
		}

		for (int i = 0; i < num_classes; i++)
			output[batch_idx * num_classes + i] /= sum;
	}
}

// Categorical Cross-Entropy loss for classification
// L = -sum(y_{o,c} * log(p_{o,c})) for each sample o and class c
__global__ void crossEntropyLossKernel(float *predictions, int *labels, float *loss, int batch_size,
									   int num_classes) {
	int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (batch_idx < batch_size) {
		int label = labels[batch_idx];
		float pred = predictions[batch_idx * num_classes + label];
		loss[batch_idx] = -logf(pred + 1e-8f); // Avoid log(0)
	}
}

// Ridge Regularization (L2) loss for weights
__global__ void ridgeL2GradientKernel(float *dW, float *W, int size, float lambda) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		dW[idx] += lambda * W[idx];
	}
}

// Combined kernel for softmax and cross-entropy loss (more numerically stable)
__global__ void softmaxCrossEntropyKernel(const float *softmax_output, const int *labels,
										  float *d_logits, int batchSize, int numClasses) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batchSize * numClasses;

	if (idx >= total)
		return;

	int b = idx / numClasses;
	int c = idx % numClasses;

	float grad = softmax_output[idx];

	if (c == labels[b])
		grad -= 1.0f;

	d_logits[idx] = grad / (float)batchSize;
}

// Get class prediction from softmax
__global__ void getPredictionsKernel(float *softmax_output, int *predictions, int batch_size,
									 int num_classes) {
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

// Backward propagation of fully connected layer
__global__ void fcWeightGradientKernel(const float *input, const float *d_logits, float *d_weights,
									   int batchSize, int inFeatures, int outClasses) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int f = blockIdx.y * blockDim.y + threadIdx.y;

	if (f >= inFeatures || c >= outClasses)
		return;

	float sum = 0.0f;

	for (int b = 0; b < batchSize; b++)
		sum += input[b * inFeatures + f] * d_logits[b * outClasses + c];

	d_weights[f * outClasses + c] = sum;
}

__global__ void fcBiasGradientKernel(const float *d_logits, float *d_bias, int batchSize,
									 int outClasses) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (c >= outClasses)
		return;

	float sum = 0.0f;

	for (int b = 0; b < batchSize; b++)
		sum += d_logits[b * outClasses + c];

	d_bias[c] = sum;
}

__global__ void fcInputGradientKernel(const float *d_logits, const float *weights, float *d_input,
									  int batchSize, int inFeatures, int outClasses) {
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	if (b >= batchSize || f >= inFeatures)
		return;

	float sum = 0.0f;

	for (int c = 0; c < outClasses; c++)
		sum += d_logits[b * outClasses + c] * weights[f * outClasses + c];

	d_input[b * inFeatures + f] = sum;
}

// Max pooling backward propagation
__global__ void maxPoolingBackwardKernel(const float *pool_input, const float *d_pool_output,
										 float *d_pool_input, int batchSize, int channels,
										 int inputSize, int poolSize, int outputSize, int stride) {
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;

	int bc = blockIdx.z;
	int c = bc % channels;
	int b = bc / channels;

	if (out_x >= outputSize || out_y >= outputSize || c >= channels || b >= batchSize)
		return;

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

	float grad = d_pool_output[b * channels * outputSize * outputSize +
							   c * outputSize * outputSize + out_y * outputSize + out_x];

	if (max_x >= 0 && max_y >= 0)
		atomicAdd(&d_pool_input[base + max_y * inputSize + max_x], grad);
}

// ReLU backward propagation
__global__ void reluBackwardKernel(const float *pre_relu, const float *d_output, float *d_input,
								   int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size)
		return;

	d_input[idx] = (pre_relu[idx] > 0.0f) ? d_output[idx] : 0.0f;
}

// Kernel convolution with NCHW layout backwardpropagation
__global__ void conv2WeightGradientReduceKernel(const float *input, const float *d_conv_output,
												float *d_kernels, int batchSize, int inputChannels,
												int inputH, int inputW, int kernelCount,
												int kernelH, int kernelW, int outputH, int outputW,
												int paddingY, int paddingX, int strideY,
												int strideX) {
	extern __shared__ float sharedSum[];

	int weightIdx = blockIdx.x;
	int tid = threadIdx.x;

	int totalWeights = kernelCount * inputChannels * kernelH * kernelW;

	if (weightIdx >= totalWeights)
		return;

	int tmp = weightIdx;

	int kx = tmp % kernelW;
	tmp /= kernelW;

	int ky = tmp % kernelH;
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
			float inputValue = input[((b * inputChannels + c) * inputH + iy) * inputW + ix];
			float gradValue = d_conv_output[((b * kernelCount + k) * outputH + oy) * outputW + ox];
			localSum += inputValue * gradValue;
		}
	}
	sharedSum[tid] = localSum;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (tid < offset)
			sharedSum[tid] += sharedSum[tid + offset];
		__syncthreads();
	}

	if (tid == 0)
		d_kernels[weightIdx] = sharedSum[0];
}

// Convolution kernel bias backpropagation
__global__ void conv2dBiasGradientReduceKernel(const float *d_conv_output, float *d_conv_bias,
											   int batchSize, int kernelCount, int outputH,
											   int outputW) {
	extern __shared__ float sharedSum[];

	int k = blockIdx.x;
	int tid = threadIdx.x;

	if (k >= kernelCount)
		return;

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
		if (tid < offset)
			sharedSum[tid] += sharedSum[tid + offset];
		__syncthreads();
	}

	if (tid == 0)
		d_conv_bias[k] = sharedSum[0];
}

// Stochastic Gradient Descent (SGD)
__global__ void sgdUpdateKernel(float *params, const float *grads, float learningRate, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		params[idx] -= learningRate * grads[idx];
}

// Classification accuracy
float calculateAccuracy(int *predictions, int *labels, int batch_size) {
	int correct = 0;
	for (int i = 0; i < batch_size; i++) {
		if (predictions[i] == labels[i])
			correct++;
	}
	return (float)correct / batch_size;
}

// Compute average cross-entropy loss for a batch
float compute_batch_loss(float *d_softmax_output, int *d_labels, float *d_loss,
								float *h_loss, int batchSize) {
	int blockSize = 256;
	int gridSize = (batchSize + blockSize - 1) / blockSize;

	crossEntropyLossKernel<<<gridSize, blockSize>>>(d_softmax_output, d_labels, d_loss, batchSize,
													NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	CHECK_CUDA_ERROR(
		cudaMemcpy(h_loss, d_loss, (size_t)batchSize * sizeof(float), cudaMemcpyDeviceToHost));

	float sum = 0.0f;

	for (int i = 0; i < batchSize; i++) {
		sum += h_loss[i];
	}

	return sum / (float)batchSize;
}
