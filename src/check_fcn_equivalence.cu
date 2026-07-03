#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "elements.h"
#include "kernels.h"
#include "lenet.h"
#include "lenet_fcn.h"

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    unsigned int seed = FIXED_SEED;
    srand(seed);

    float *h_input = (float *)malloc((size_t)inputElements * sizeof(float));
    int *h_labels = (int *)calloc((size_t)BATCH_SIZE, sizeof(int));
    float *h_logits_fc = (float *)malloc((size_t)logitsElements * sizeof(float));
    float *h_softmax_fc = (float *)malloc((size_t)logitsElements * sizeof(float));
    float *h_logits_fcn = (float *)malloc((size_t)logitsElements * sizeof(float));
    float *h_softmax_fcn = (float *)malloc((size_t)logitsElements * sizeof(float));
    int *h_pred_fc = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
    int *h_pred_fcn = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));

    if (!h_input || !h_labels || !h_logits_fc || !h_softmax_fc || !h_logits_fcn ||
        !h_softmax_fcn || !h_pred_fc || !h_pred_fcn) {
        fprintf(stderr, "Host malloc failed.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < inputElements; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }

    float *d_input = NULL;
    int *d_labels = NULL;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, (size_t)inputElements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, (size_t)BATCH_SIZE * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, (size_t)inputElements * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels, (size_t)BATCH_SIZE * sizeof(int),
                                cudaMemcpyHostToDevice));

    LeNet *cnn = LeNet_init(seed);
    LeNetFCN *fcn = LeNetFCN_wrap(cnn);

    LeNet_forward(d_input, d_labels, cnn, NULL);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_logits_fc, cnn->d_logits,
                                (size_t)logitsElements * sizeof(float),
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_softmax_fc, cnn->d_softmax_output,
                                (size_t)logitsElements * sizeof(float),
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pred_fc, cnn->d_predictions,
                                (size_t)BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    LeNetFCN_forward(d_input, d_labels, fcn, NULL);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_logits_fcn, cnn->d_logits,
                                (size_t)logitsElements * sizeof(float),
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_softmax_fcn, cnn->d_softmax_output,
                                (size_t)logitsElements * sizeof(float),
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pred_fcn, cnn->d_predictions,
                                (size_t)BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    int pred_mismatches = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (h_pred_fc[i] != h_pred_fcn[i]) pred_mismatches++;
    }

    printf("FC vs FCN equivalence check\n");
    printf("Max |logits_fc - logits_fcn|:   %.9g\n",
           max_abs_diff(h_logits_fc, h_logits_fcn, logitsElements));
    printf("Max |softmax_fc - softmax_fcn|: %.9g\n",
           max_abs_diff(h_softmax_fc, h_softmax_fcn, logitsElements));
    printf("Prediction mismatches: %d/%d\n", pred_mismatches, BATCH_SIZE);

    LeNetFCN_free(fcn);
    LeNet_free(cnn);
    cudaFree(d_input);
    cudaFree(d_labels);

    free(h_input);
    free(h_labels);
    free(h_logits_fc);
    free(h_softmax_fc);
    free(h_logits_fcn);
    free(h_softmax_fcn);
    free(h_pred_fc);
    free(h_pred_fcn);

    return pred_mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
