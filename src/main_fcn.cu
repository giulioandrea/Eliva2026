#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "elements.h"
#include "kernels.h"
#include "lenet.h"
#include "lenet_fcn.h"

/*
 * Additive-only executable that leaves main.cu unchanged.
 * It trains/evaluates the same model but replaces the forward classification head:
 *   Flatten + FC
 * with the equivalent:
 *   Conv2D(kernel=POOL_OUTPUT_H x POOL_OUTPUT_W, filters=NUM_CLASSES)
 * using reshaped FC weights.
 */
static const int BATCH_PRINT_EVERY = 10;
static const int ENABLE_TIMING = 1;

/* timing[0] = convolution
 * timing[1] = ReLU
 * timing[2] = max pooling
 * timing[3] = FC-as-Conv2D + bias
 * timing[4] = softmax + prediction
 */
static void print_forward_timing(const char *phase, int epoch, int batch, const float t[5]) {
    float total = t[0] + t[1] + t[2] + t[3] + t[4];

    printf("Timing | %s | Epoch %d | Batch %d | "
           "Forward %.3f ms | Conv %.3f | ReLU %.3f | Pool %.3f | "
           "FC-as-Conv+Bias %.3f | Softmax+Pred %.3f\n",
           phase, epoch, batch, total, t[0], t[1], t[2], t[3], t[4]);
    fflush(stdout);
}

static void print_backward_timing(const char *phase, int epoch, int batch, float t) {
    printf("Timing | %s | Epoch %d | Batch %d | Backward %.3f ms\n", phase, epoch, batch, t);
    fflush(stdout);
}

static int should_print_batch(int batch, int totalBatches) {
    if (BATCH_PRINT_EVERY <= 0) return 0;

    int currentBatch = batch + 1;
    return ((currentBatch % BATCH_PRINT_EVERY) == 0) || (currentBatch == totalBatches);
}

static int should_collect_timing(int batch, int totalBatches) {
    return ENABLE_TIMING && should_print_batch(batch, totalBatches);
}

int main(int argc, char **argv) {
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

    if (testBatches <= 0) {
        fprintf(stderr, "Test dataset too small for BATCH_SIZE=%d.\n", BATCH_SIZE);
        free_dataset(&train);
        free_dataset(&test);
        return EXIT_FAILURE;
    }

    printf("\nNetwork configuration: FC-as-Conv2D forward path\n");
    printf("Input:        %d x %d x %d\n", INPUT_CHANNELS, INPUT_H, INPUT_W);
    printf("Conv output:  %d x %d x %d\n", KERNEL_COUNT, OUTPUT_H, OUTPUT_W);
    printf("Pool output:  %d x %d x %d\n", KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W);
    printf("Original FC:  %d x %d\n", FLATTEN_SIZE, NUM_CLASSES);
    printf("FCN head:     %d filters of shape %d x %d x %d -> 1 x 1 logits\n",
           NUM_CLASSES, KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W);
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
    LeNetFCN *fcn = LeNetFCN_wrap(cnn);

    cudaEvent_t backwardStart;
    cudaEvent_t backwardStop;
    CHECK_CUDA_ERROR(cudaEventCreate(&backwardStart));
    CHECK_CUDA_ERROR(cudaEventCreate(&backwardStop));

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

            if (loaded != BATCH_SIZE) continue;

            CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, (size_t)inputElements * sizeof(float),
                                        cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels, (size_t)BATCH_SIZE * sizeof(int),
                                        cudaMemcpyHostToDevice));

            int printBatch = should_print_batch(batch, trainBatches);
            int collectTiming = should_collect_timing(batch, trainBatches);
            float forwardTiming[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float backwardTiming = 0.0f;

            // Safe FCN forward: syncs reshaped Conv2D weights from the current FC weights.
            LeNetFCN_forward(d_input, d_labels, fcn, collectTiming ? forwardTiming : NULL);
            CHECK_KERNEL_LAUNCH();

            // Keep the original backward path intact. It is mathematically equivalent because the
            // forward logits are identical to Flatten+FC up to floating-point reduction order.
            if (collectTiming) {
                CHECK_CUDA_ERROR(cudaEventRecord(backwardStart, 0));
                LeNet_backward(d_input, d_labels, cnn, learningRate, lambda, NULL);
                CHECK_CUDA_ERROR(cudaEventRecord(backwardStop, 0));
                CHECK_CUDA_ERROR(cudaEventSynchronize(backwardStop));
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&backwardTiming, backwardStart, backwardStop));
            } else {
                LeNet_backward(d_input, d_labels, cnn, learningRate, lambda, NULL);
            }

            float batchLoss =
                compute_batch_loss(cnn->d_softmax_output, d_labels, d_loss, h_loss, BATCH_SIZE);

            CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, cnn->d_predictions,
                                        (size_t)BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

            float batchAccuracy = calculateAccuracy(h_predictions, h_labels, BATCH_SIZE);

            trainLossSum += batchLoss;
            trainAccuracySum += batchAccuracy;
            usedTrainBatches++;

            if (printBatch) {
                printf("Epoch %d/%d | Batch %d/%d | Train loss %.4f | Train acc %.4f\n",
                       epoch + 1, EPOCHS, batch + 1, trainBatches, batchLoss, batchAccuracy);

                if (collectTiming) {
                    print_forward_timing("Train-FCN", epoch + 1, batch + 1, forwardTiming);
                    print_backward_timing("Train-FCN", epoch + 1, batch + 1, backwardTiming);
                }
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

        // No weights change during test, so one reshape is enough for the entire test pass.
        LeNetFCN_sync_from_fc(fcn);

        for (int batch = 0; batch < testBatches; batch++) {
            int startIndex = batch * BATCH_SIZE;
            int loaded = load_batch(&test, startIndex, BATCH_SIZE, h_input, h_labels);

            if (loaded != BATCH_SIZE) continue;

            CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, (size_t)inputElements * sizeof(float),
                                        cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels, (size_t)BATCH_SIZE * sizeof(int),
                                        cudaMemcpyHostToDevice));

            int printBatch = should_print_batch(batch, testBatches);
            int collectTiming = should_collect_timing(batch, testBatches);
            float forwardTiming[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            LeNetFCN_forward_no_sync(d_input, d_labels, fcn,
                                     collectTiming ? forwardTiming : NULL);
            CHECK_KERNEL_LAUNCH();

            float batchLoss =
                compute_batch_loss(cnn->d_softmax_output, d_labels, d_loss, h_loss, BATCH_SIZE);

            CHECK_CUDA_ERROR(cudaMemcpy(h_predictions, cnn->d_predictions,
                                        (size_t)BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

            float batchAccuracy = calculateAccuracy(h_predictions, h_labels, BATCH_SIZE);

            testLossSum += batchLoss;
            testAccuracySum += batchAccuracy;
            usedTestBatches++;

            if (printBatch) {
                printf("Epoch %d/%d | Batch %d/%d | Test loss %.4f | Test acc %.4f\n",
                       epoch + 1, EPOCHS, batch + 1, testBatches, batchLoss, batchAccuracy);

                if (collectTiming) {
                    print_forward_timing("Test-FCN", epoch + 1, batch + 1, forwardTiming);
                }
            }
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

    CHECK_CUDA_ERROR(cudaEventDestroy(backwardStart));
    CHECK_CUDA_ERROR(cudaEventDestroy(backwardStop));

    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_loss);

    free(h_input);
    free(h_labels);
    free(h_predictions);
    free(h_loss);

    LeNetFCN_free(fcn);
    LeNet_free(cnn);

    free_dataset(&train);
    free_dataset(&test);

    return EXIT_SUCCESS;
}
