#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "kernels.h"

int main(int argc, char **argv) {
	// Uncomment to use getrandom for better randomness in shuffling
	// unsigned char buffer[16];
	// getrandom(buffer, sizeof(buffer), 0);
	// srand(*(unsigned int *)buffer);

	// Comment out to ensure reproducibility for testing and benchmarking
	srand(FIXED_SEED);

	const char *datasetRoot = (argc > 1) ? argv[1] : "dataset";

	char trainPath[4096];
	char testPath[4096];

	snprintf(trainPath, sizeof(trainPath), "%s/train", datasetRoot);
	snprintf(testPath, sizeof(testPath), "%s/test", datasetRoot);

	Dataset train;
	Dataset test;

	if (!load_dataset_index(&train, trainPath)) {
		fprintf(stderr, "Failed to load training dataset from '%s'.\n", trainPath);
		return EXIT_FAILURE;
	}

	if (!load_dataset_index(&test, testPath)) {
		fprintf(stderr, "Failed to load test dataset from '%s'.\n", testPath);
		free_dataset(&train);
		return EXIT_FAILURE;
	}

	printf("\nTRAIN DATASET\n");
	print_dataset_info(&train);

	printf("\nTEST DATASET\n");
	print_dataset_info(&test);

	int trainBatches = train.count / BATCH_SIZE;
	int testBatches = test.count / BATCH_SIZE;

	if (trainBatches <= 0) {
		fprintf(stderr, "Training dataset too small for BATCH_SIZE=%d.\n", BATCH_SIZE);
		free_dataset(&train);
		free_dataset(&test);
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

	const int inputElements = BATCH_SIZE * INPUT_CHANNELS * INPUT_H * INPUT_W;

	const int convElements = BATCH_SIZE * KERNEL_COUNT * OUTPUT_H * OUTPUT_W;

	const int poolElements = BATCH_SIZE * KERNEL_COUNT * POOL_OUTPUT_H * POOL_OUTPUT_W;

	const int logitsElements = BATCH_SIZE * NUM_CLASSES;

	const int kernelElements = KERNEL_COUNT * INPUT_CHANNELS * KERNEL_H * KERNEL_W;

	const int fcWeightElements = FLATTEN_SIZE * NUM_CLASSES;

	float *h_input = (float *)malloc((size_t)inputElements * sizeof(float));
	int *h_labels = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
	int *h_predictions = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
	float *h_loss = (float *)malloc((size_t)BATCH_SIZE * sizeof(float));

	float *h_kernels = (float *)malloc((size_t)kernelElements * sizeof(float));
	float *h_fc_weights = (float *)malloc((size_t)fcWeightElements * sizeof(float));
	float *h_fc_bias = (float *)malloc((size_t)NUM_CLASSES * sizeof(float));

	if (!h_input || !h_labels || !h_predictions || !h_loss || !h_kernels || !h_fc_weights ||
		!h_fc_bias) {
		fprintf(stderr, "Host malloc failed.\n");

		free(h_input);
		free(h_labels);
		free(h_predictions);
		free(h_loss);
		free(h_kernels);
		free(h_fc_weights);
		free(h_fc_bias);

		free_dataset(&train);
		free_dataset(&test);

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
	CHECK_CUDA_ERROR(
		cudaMalloc((void **)&d_softmax_output, (size_t)logitsElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, (size_t)BATCH_SIZE * sizeof(int)));

	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_logits, (size_t)logitsElements * sizeof(float)));
	CHECK_CUDA_ERROR(
		cudaMalloc((void **)&d_d_fc_weights, (size_t)fcWeightElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_fc_bias, (size_t)NUM_CLASSES * sizeof(float)));
	CHECK_CUDA_ERROR(
		cudaMalloc((void **)&d_d_pooling_output, (size_t)poolElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_activation, (size_t)convElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_conv_output, (size_t)convElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_d_kernels, (size_t)kernelElements * sizeof(float)));

	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_loss, (size_t)BATCH_SIZE * sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_kernels, h_kernels, (size_t)kernelElements * sizeof(float),
								cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMemcpy(d_fc_weights, h_fc_weights,
								(size_t)fcWeightElements * sizeof(float), cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMemcpy(d_fc_bias, h_fc_bias, (size_t)NUM_CLASSES * sizeof(float),
								cudaMemcpyHostToDevice));

	const float learningRate = 0.001f;
	const float lambda = 1e-4f;

	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		shuffle_dataset(&train);

		float trainLossSum = 0.0f;
		float trainAccuracySum = 0.0f;
		int usedTrainBatches = 0;

		for (int batch = 0; batch < trainBatches; batch++) {
			int startIndex = batch * BATCH_SIZE;

			int loaded = load_batch(&train, startIndex, BATCH_SIZE, h_input, h_labels);

			if (loaded != BATCH_SIZE) {
				continue;
			}

			CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, (size_t)inputElements * sizeof(float),
										cudaMemcpyHostToDevice));

			CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels, (size_t)BATCH_SIZE * sizeof(int),
										cudaMemcpyHostToDevice));

			trainBatch(d_input, d_labels, d_kernels, d_conv_output, d_activation, d_pooling_output,
					   d_fc_weights, d_fc_bias, d_logits, d_softmax_output, d_predictions,
					   d_d_logits, d_d_fc_weights, d_d_fc_bias, d_d_pooling_output, d_d_activation,
					   d_d_conv_output, d_d_kernels, learningRate, lambda);

			float batchLoss =
				compute_batch_loss(d_softmax_output, d_labels, d_loss, h_loss, BATCH_SIZE);

			CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, d_predictions,
										(size_t)BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

			float batchAccuracy = calculateAccuracy(h_predictions, h_labels, BATCH_SIZE);

			trainLossSum += batchLoss;
			trainAccuracySum += batchAccuracy;
			usedTrainBatches++;

			if ((batch + 1) % 10 == 0) {
				printf("Epoch %d/%d | Batch %d/%d | Train loss %.4f | Train acc %.4f\n", epoch + 1,
					   EPOCHS, batch + 1, trainBatches, batchLoss, batchAccuracy);
			}
		}

		float avgTrainLoss = 0.0f;
		float avgTrainAccuracy = 0.0f;

		if (usedTrainBatches > 0) {
			avgTrainLoss = trainLossSum / (float)usedTrainBatches;
			avgTrainAccuracy = trainAccuracySum / (float)usedTrainBatches;
		}

		float testLossSum = 0.0f;
		float testAccuracySum = 0.0f;
		int usedTestBatches = 0;

		for (int batch = 0; batch < testBatches; batch++) {
			int startIndex = batch * BATCH_SIZE;

			int loaded = load_batch(&test, startIndex, BATCH_SIZE, h_input, h_labels);

			if (loaded != BATCH_SIZE) {
				continue;
			}

			CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, (size_t)inputElements * sizeof(float),
										cudaMemcpyHostToDevice));

			CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels, (size_t)BATCH_SIZE * sizeof(int),
										cudaMemcpyHostToDevice));

			float timing[5] = {0.0f};

			forwardCNNClassifier(d_input, d_kernels, d_conv_output, d_activation, d_pooling_output,
								 d_fc_weights, d_fc_bias, d_logits, d_softmax_output, d_predictions,
								 timing);

			CHECK_CUDA_ERROR(cudaDeviceSynchronize());

			float batchLoss =
				compute_batch_loss(d_softmax_output, d_labels, d_loss, h_loss, BATCH_SIZE);

			CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, d_predictions,
										(size_t)BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

			float batchAccuracy = calculateAccuracy(h_predictions, h_labels, BATCH_SIZE);

			testLossSum += batchLoss;
			testAccuracySum += batchAccuracy;
			usedTestBatches++;
		}

		float avgTestLoss = 0.0f;
		float avgTestAccuracy = 0.0f;

		if (usedTestBatches > 0) {
			avgTestLoss = testLossSum / (float)usedTestBatches;
			avgTestAccuracy = testAccuracySum / (float)usedTestBatches;
		}

		printf("\nEpoch %d/%d completed | "
			   "Train loss %.4f | Train acc %.4f | Used train batches %d/%d | "
			   "Test loss %.4f | Test acc %.4f | Used test batches %d/%d\n\n",
			   epoch + 1, EPOCHS, avgTrainLoss, avgTrainAccuracy, usedTrainBatches, trainBatches,
			   avgTestLoss, avgTestAccuracy, usedTestBatches, testBatches);
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
	free_dataset(&test);

	return EXIT_SUCCESS;
}
