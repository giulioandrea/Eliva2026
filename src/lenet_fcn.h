#ifndef LENET_FCN_H
#define LENET_FCN_H

#include "lenet.h"

typedef struct {
    // Non-owning pointer. The original LeNet object remains allocated/freed by the caller.
    LeNet *base;

    // Additive FC-as-Conv2D view of base->d_fc_weights.
    // Shape: [NUM_CLASSES, KERNEL_COUNT, POOL_OUTPUT_H, POOL_OUTPUT_W].
    float *d_fc_conv_kernels;
} LeNetFCN;

LeNetFCN *LeNetFCN_wrap(LeNet *base);

void LeNetFCN_free(LeNetFCN *fcn);

// Rebuild d_fc_conv_kernels from base->d_fc_weights. Call this after FC weights change.
void LeNetFCN_sync_from_fc(LeNetFCN *fcn);

// Safe forward path: sync FC weights, then run the fully-convolutional forward pass.
void LeNetFCN_forward(float *d_input, int *d_labels, LeNetFCN *fcn, float *timing);

// Faster inference path: assumes LeNetFCN_sync_from_fc() has already been called and
// base->d_fc_weights have not changed since then.
void LeNetFCN_forward_no_sync(float *d_input, int *d_labels, LeNetFCN *fcn, float *timing);

#endif // LENET_FCN_H
