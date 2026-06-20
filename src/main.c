#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "kernels.h"

#include "elements.h"
#include "lenet.h"

int main(int argc, char **argv) {
	// Uncomment to use getrandom for better randomness in shuffling
	// unsigned char buffer[16];
	// getrandom(buffer, sizeof(buffer), 0);
	// unsigned int seed = (*(unsigned int *)buffer);
	unsigned int seed = FIXED_SEED;

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

	float *h_input = (float *)malloc((size_t)inputElements * sizeof(float));
	int *h_labels = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
	int *h_predictions = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
	float *h_loss = (float *)malloc((size_t)BATCH_SIZE * sizeof(float));

	if (!h_input || !h_labels || !h_predictions || !h_loss) {
		fprintf(stderr, "Host malloc failed.\n");

		free(h_input);
		free(h_labels);
		free(h_predictions);
		free(h_loss);

		free_dataset(&train);
		free_dataset(&test);

		return EXIT_FAILURE;
	}

	float *d_input = NULL;
	int *d_labels = NULL;
	float *d_loss = NULL;

	CHECK_CUDA_ERROR(cudaMalloc(&d_input, (size_t)inputElements * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_labels, (size_t)BATCH_SIZE * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_loss, (size_t)BATCH_SIZE * sizeof(float)));

	LeNet *cnn = LeNet_init(seed);

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

			float timing[5] = {
				0.0f}; // TODO: this is just temporary, and needs work (possible seg-fault)

			LeNet_forward(d_input, d_labels, cnn, timing);
			LeNet_backward(d_input, d_labels, cnn, learningRate, lambda, timing);

			CHECK_CUDA_ERROR(cudaDeviceSynchronize());

			float batchLoss =
				compute_batch_loss(cnn->d_softmax_output, d_labels, d_loss, h_loss, BATCH_SIZE);

			CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, cnn->d_predictions,
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

			float timing[5] = {
				0.0f}; // TODO: this is just temporary, and needs work (possible seg-fault)

			LeNet_forward(d_input, d_labels, cnn, timing);

			CHECK_CUDA_ERROR(cudaDeviceSynchronize());

			float batchLoss =
				compute_batch_loss(cnn->d_softmax_output, d_labels, d_loss, h_loss, BATCH_SIZE);

			CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, cnn->d_predictions,
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
	cudaFree(d_loss);

	free(h_input);
	free(h_labels);
	free(h_predictions);
	free(h_loss);

	LeNet_free(cnn);

	free_dataset(&train);
	free_dataset(&test);

	return EXIT_SUCCESS;
}
