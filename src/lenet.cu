#include "elements.h"
#include "kernels.h"
#include "lenet.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

LeNet *LeNet_init(unsigned int seed) {
	LeNet *cnn = (LeNet *)malloc(sizeof(LeNet));

	float *h_kernels = (float *)malloc((size_t)kernelElements * sizeof(float));
	float *h_fc_weights = (float *)malloc((size_t)fcWeightElements * sizeof(float));
	float *h_fc_bias = (float *)malloc((size_t)NUM_CLASSES * sizeof(float));

	if (!h_kernels || !h_fc_weights || !h_fc_bias) {
		fprintf(stderr, "Host malloc failed.\n");
		exit(EXIT_FAILURE);
	}

	srand(seed);

	initializeKernels(h_kernels);
	initializeFullyConnected(h_fc_weights, h_fc_bias);

	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_kernels, (size_t)kernelElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_conv_output, (size_t)convElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_activation, (size_t)convElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_pooling_output, (size_t)poolElements * sizeof(float)));

	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_fc_weights, (size_t)fcWeightElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_fc_bias, (size_t)NUM_CLASSES * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_logits, (size_t)logitsElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_softmax_output, (size_t)logitsElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_predictions, (size_t)BATCH_SIZE * sizeof(int)));

	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_logits, (size_t)logitsElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_fc_weights, (size_t)fcWeightElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_fc_bias, (size_t)NUM_CLASSES * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_pooling_output, (size_t)poolElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_activation, (size_t)convElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_conv_output, (size_t)convElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&cnn->d_d_kernels, (size_t)kernelElements * sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(cnn->d_kernels, h_kernels, (size_t)kernelElements * sizeof(float),
								cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMemcpy(cnn->d_fc_weights, h_fc_weights,
								(size_t)fcWeightElements * sizeof(float), cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMemcpy(cnn->d_fc_bias, h_fc_bias, (size_t)NUM_CLASSES * sizeof(float),
								cudaMemcpyHostToDevice));

	free(h_kernels);
	free(h_fc_weights);
	free(h_fc_bias);

	return cnn;
}

void LeNet_free(LeNet *cnn) {
	cudaFree(cnn->d_kernels);
	cudaFree(cnn->d_conv_output);
	cudaFree(cnn->d_activation);
	cudaFree(cnn->d_pooling_output);

	cudaFree(cnn->d_fc_weights);
	cudaFree(cnn->d_fc_bias);
	cudaFree(cnn->d_logits);
	cudaFree(cnn->d_softmax_output);
	cudaFree(cnn->d_predictions);

	cudaFree(cnn->d_d_logits);
	cudaFree(cnn->d_d_fc_weights);
	cudaFree(cnn->d_d_fc_bias);
	cudaFree(cnn->d_d_pooling_output);
	cudaFree(cnn->d_d_activation);
	cudaFree(cnn->d_d_conv_output);
	cudaFree(cnn->d_d_kernels);

	free(cnn);
}

void LeNet_forward(float *d_input, int *d_labels, LeNet *cnn, float *timing) {
	const bool collectTiming = timing != NULL;
	cudaEvent_t start, stop;

	if (collectTiming) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	// Convolution
	dim3 convBlockDim(8, 8);
	dim3 convGridDim((OUTPUT_SIZE + convBlockDim.x - 1) / convBlockDim.x,
					 (OUTPUT_SIZE + convBlockDim.y - 1) / convBlockDim.y,
					 BATCH_SIZE * KERNEL_COUNT);

	int tileSize = convBlockDim.x;
	int tileSizeWithPadding = tileSize + KERNEL_SIZE - 1;
	int sharedMemSize = INPUT_CHANNELS * tileSizeWithPadding * tileSizeWithPadding * sizeof(float);
	if (collectTiming) {
		cudaEventRecord(start);
	}

	convolutionSharedKernel<<<convGridDim, convBlockDim, sharedMemSize>>>(
		d_input, cnn->d_kernels, cnn->d_conv_output, BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
		KERNEL_SIZE, KERNEL_COUNT, OUTPUT_SIZE, PADDING, STRIDE);

	if (collectTiming) {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing[0], start, stop);
	}

	// Relu
	int convTotalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
	cudaMemcpy(cnn->d_activation, cnn->d_conv_output, convTotalElements * sizeof(float),
			   cudaMemcpyDeviceToDevice);

	int blockSize = 256;
	int reluGridSize = (convTotalElements + blockSize - 1) / blockSize;

	if (collectTiming) {
		cudaEventRecord(start);
	}
	reluActivationKernel<<<reluGridSize, blockSize>>>(cnn->d_activation, convTotalElements);

	if (collectTiming) {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing[1], start, stop);
	}

	// Max pooling
	dim3 poolBlockDim(8, 8);
	dim3 poolGridDim((POOL_OUTPUT_SIZE + poolBlockDim.x - 1) / poolBlockDim.x,
					 (POOL_OUTPUT_SIZE + poolBlockDim.y - 1) / poolBlockDim.y,
					 BATCH_SIZE * KERNEL_COUNT);

	if (collectTiming) {
		cudaEventRecord(start);
	}

	maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(cnn->d_activation, cnn->d_pooling_output,
													BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
													POOL_SIZE, POOL_OUTPUT_SIZE, POOL_STRIDE);

	if (collectTiming) {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing[2], start, stop);
	}

	// Fully connected classification: logits = pooling_output * fc_weights.
	dim3 mmBlockDim(TILE_SIZE, TILE_SIZE);
	dim3 mmGridDim((NUM_CLASSES + TILE_SIZE - 1) / TILE_SIZE,
				   (BATCH_SIZE + TILE_SIZE - 1) / TILE_SIZE);

	if (collectTiming) {
		cudaEventRecord(start);
	}

	matrixMultiplySharedKernel<<<mmGridDim, mmBlockDim>>>(cnn->d_pooling_output, cnn->d_fc_weights,
														  cnn->d_logits, BATCH_SIZE, FLATTEN_SIZE,
														  NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	// Add bias
	dim3 biasBlockDim(16, 16);
	dim3 biasGridDim((NUM_CLASSES + biasBlockDim.x - 1) / biasBlockDim.x,
					 (BATCH_SIZE + biasBlockDim.y - 1) / biasBlockDim.y);

	addBiasKernel<<<biasGridDim, biasBlockDim>>>(cnn->d_logits, cnn->d_fc_bias, BATCH_SIZE,
												 NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	if (collectTiming) {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing[3], start, stop);
	}

	// Softmax
	int predBlockSize = 256;
	int predGridSize = (BATCH_SIZE + predBlockSize - 1) / predBlockSize;

	if (collectTiming) {
		cudaEventRecord(start);
	}

	softmaxKernel<<<predGridSize, predBlockSize>>>(cnn->d_logits, cnn->d_softmax_output, BATCH_SIZE,
												   NUM_CLASSES);

	// Prediction
	getPredictionsKernel<<<predGridSize, predBlockSize>>>(cnn->d_softmax_output, cnn->d_predictions,
														  BATCH_SIZE, NUM_CLASSES);

	if (collectTiming) {
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timing[4], start, stop);
	}

	if (collectTiming) {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

// Backpropagation
void LeNet_backward(float *d_input, int *d_labels, LeNet *cnn, float learningRate, float lambda,
					float *timing) {
	int blockSize = 256;

	int logitsSize = BATCH_SIZE * NUM_CLASSES;
	int logitsGrid = (logitsSize + blockSize - 1) / blockSize;

	softmaxCrossEntropyKernel<<<logitsGrid, blockSize>>>(cnn->d_softmax_output, d_labels,
														 cnn->d_d_logits, BATCH_SIZE, NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	dim3 fcWeightBlock(16, 16);
	dim3 fcWeightGrid((NUM_CLASSES + fcWeightBlock.x - 1) / fcWeightBlock.x,
					  (FLATTEN_SIZE + fcWeightBlock.y - 1) / fcWeightBlock.y);

	fcWeightGradientKernel<<<fcWeightGrid, fcWeightBlock>>>(cnn->d_pooling_output, cnn->d_d_logits,
															cnn->d_d_fc_weights, BATCH_SIZE,
															FLATTEN_SIZE, NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	int biasGrid = (NUM_CLASSES + blockSize - 1) / blockSize;

	fcBiasGradientKernel<<<biasGrid, blockSize>>>(cnn->d_d_logits, cnn->d_d_fc_bias, BATCH_SIZE,
												  NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	dim3 fcInputBlock(16, 16);
	dim3 fcInputGrid((FLATTEN_SIZE + fcInputBlock.x - 1) / fcInputBlock.x,
					 (BATCH_SIZE + fcInputBlock.y - 1) / fcInputBlock.y);

	fcInputGradientKernel<<<fcInputGrid, fcInputBlock>>>(cnn->d_d_logits, cnn->d_fc_weights,
														 cnn->d_d_pooling_output, BATCH_SIZE,
														 FLATTEN_SIZE, NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	// d_activation reset for max pooling backward
	int convTotalElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_SIZE * OUTPUT_SIZE;
	CHECK_CUDA_ERROR(cudaMemset(cnn->d_d_activation, 0, convTotalElements * sizeof(float)));

	dim3 poolBlockDim(8, 8);
	dim3 poolGridDim((POOL_OUTPUT_SIZE + poolBlockDim.x - 1) / poolBlockDim.x,
					 (POOL_OUTPUT_SIZE + poolBlockDim.y - 1) / poolBlockDim.y,
					 BATCH_SIZE * KERNEL_COUNT);

	maxPoolingBackwardKernel<<<poolGridDim, poolBlockDim>>>(
		cnn->d_activation, cnn->d_d_pooling_output, cnn->d_d_activation, BATCH_SIZE, KERNEL_COUNT,
		OUTPUT_SIZE, POOL_SIZE, POOL_OUTPUT_SIZE, POOL_STRIDE);
	CHECK_KERNEL_LAUNCH();

	// ReLU backward
	int reluGrid = (convTotalElements + blockSize - 1) / blockSize;

	reluBackwardKernel<<<reluGrid, blockSize>>>(cnn->d_conv_output, // pre-ReLU values
												cnn->d_d_activation, cnn->d_d_conv_output,
												convTotalElements);
	CHECK_KERNEL_LAUNCH();

	int convWeight = KERNEL_COUNT * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
	int convWGrid = convWeight;

	conv2WeightGradientReduceKernel<<<convWGrid, blockSize, blockSize * sizeof(float)>>>(
		d_input, cnn->d_d_conv_output, cnn->d_d_kernels, BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE,
		INPUT_SIZE, KERNEL_COUNT, KERNEL_SIZE, KERNEL_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, PADDING,
		PADDING, STRIDE, STRIDE);
	CHECK_KERNEL_LAUNCH();

	int fcWeights = FLATTEN_SIZE * NUM_CLASSES;
	int fcUpdateGrid = (fcWeights + blockSize - 1) / blockSize;

	ridgeL2GradientKernel<<<fcUpdateGrid, blockSize>>>(cnn->d_d_fc_weights, cnn->d_fc_weights,
													   fcWeights, lambda);
	CHECK_KERNEL_LAUNCH();

	sgdUpdateKernel<<<fcUpdateGrid, blockSize>>>(cnn->d_fc_weights, cnn->d_d_fc_weights,
												 learningRate, fcWeights);
	CHECK_KERNEL_LAUNCH();

	int fcBiasGrid = (NUM_CLASSES + blockSize - 1) / blockSize;
	sgdUpdateKernel<<<fcBiasGrid, blockSize>>>(cnn->d_fc_bias, cnn->d_d_fc_bias, learningRate,
											   NUM_CLASSES);
	CHECK_KERNEL_LAUNCH();

	int convWeightGrid = (convWeight + blockSize - 1) / blockSize;

	ridgeL2GradientKernel<<<convWeightGrid, blockSize>>>(cnn->d_d_kernels, cnn->d_kernels,
														 convWeight, lambda);
	CHECK_KERNEL_LAUNCH();

	sgdUpdateKernel<<<convWeightGrid, blockSize>>>(cnn->d_kernels, cnn->d_d_kernels, learningRate,
												   convWeight);
	CHECK_KERNEL_LAUNCH();
}
