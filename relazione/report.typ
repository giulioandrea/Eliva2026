#set document(
  title: "CUDA CNN Report on CIFAR-10",
  author: "Bertetto Luca, Quaglia Giulio Andrea",
)

#set page(
  paper: "a4",
  margin: (x: 4cm, top: 4cm, bottom: 4cm),
  numbering: "1",
)
#set text(
  font: "New Computer Modern",
  size: 11pt,
  hyphenate: true,
)
#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 1.5em,
  spacing: 1.2em,
)
#set heading(numbering: (..nums) => (
  numbering("1.1", ..nums) + h(0.6em)
))

// Import modern plotting packages
#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot

// code block setup
#import "@preview/codly:1.3.0": *

#show: codly-init.with()
#codly(
  zebra-fill: none,
  fill: rgb("f2f2eb"),
  number-placement: "outside",

  // Set custom font and color for the line numbers
  number-format: n => text(
    font: "New Computer Modern Mono",
    fill: rgb("888888"),
    size: 9pt, // Optional: Adjust size if needed
  )[#n],
)
#show raw: set text(font: "New Computer Modern Mono", size: 9pt)
#show figure.where(kind: raw): set block(breakable: true)


// appendices
#let appendix(body) = {
  counter(heading).update(0)
  set heading(numbering: "A.1", supplement: [Appendix])
  body
}

#align(center)[
  #text(
    size: 20pt,
    weight: "bold",
  )[A CUDA Convolutional Network for CIFAR-10 Image Classification] \
  #v(0.3cm)
  #text(
    size: 12pt,
  )[Report focused on image-processing aspects of the implemented net] \
  #v(0.2cm)
  #text(size: 10pt)[26 June 2026]
]

#v(0.8cm)
#line(length: 100%)
#v(0.5cm)

= Abstract

This report analyses a CUDA/C implementation of a shallow convolutional neural network trained on CIFAR-10. The network receives a low-resolution RGB image, applies learned spatial convolution filters, introduces non-linearity with ReLU, reduces local spatial variation with max pooling, and classifies the resulting feature vector with a fully connected softmax head. The run used all 50,000 training samples and all 10,000 test samples. Over ten epochs, the test accuracy increased from 32.85% to 51.75%, while the test loss decreased from 1.9355 to 1.3848. The final train-test gap was small, approximately 0.51 percentage points, which indicates that the principal limitation is model capacity and representation rather than overfitting. A Python/Keras LeNet baseline is also compared: it reaches 58.89% test accuracy after ten epochs, but it uses a deeper architecture, Adam, and in-memory CIFAR-10 arrays. A fully convolutional equivalent of the CUDA classifier is then evaluated: reshaping the $8192 times 10$ fully connected matrix into ten $32 times 16 times 16$ valid-convolution filters preserves predictions up to floating-point reduction order, with zero prediction mismatches on the equivalence check, and reduces mean logged training forward time from 8.164 ms to 0.989 ms in the first CUDA timing run. The same original and FC-as-Conv2D executables were also run on an NVIDIA Jetson. On that run, the final test accuracy was 52.41% for both classifier heads, while the FC-as-Conv2D forward path reduced mean logged training forward time from 0.638 ms to 0.193 ms and reduced the classifier-head timing from 0.470 ms to 0.027 ms. The pure CUDA implementation is therefore best interpreted as a transparent custom-kernel baseline whose end-to-end bottleneck is dataset reading and decoding from individual image files, while the fully convolutional head removes the measured GPU forward bottleneck caused by the original custom FC kernel on both measured platforms.

#pagebreak()
#outline(title: [Contents])
#pagebreak()

= Dataset: CIFAR-10 as an image-processing benchmark

CIFAR-10 is a standard small-image benchmark from the University of Toronto @krizhevsky_cifar. The official dataset description states that it contains 60,000 RGB colour images of size $32 times 32$, organized into ten mutually exclusive object classes with 6,000 images per class. The standard split contains 50,000 training images and 10,000 test images. The official binary and Python/Matlab layouts store each image as 3072 channel values: 1024 red, 1024 green, and 1024 blue pixels.

#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 5pt,
  align: (center, left, right, right),
  [*Label*], [*Class*], [*Train samples*], [*Test samples*],
  [0], [airplane], [5000], [1000],
  [1], [automobile], [5000], [1000],
  [2], [bird], [5000], [1000],
  [3], [cat], [5000], [1000],
  [4], [deer], [5000], [1000],
  [5], [dog], [5000], [1000],
  [6], [frog], [5000], [1000],
  [7], [horse], [5000], [1000],
  [8], [ship], [5000], [1000],
  [9], [truck], [5000], [1000],
), caption: [Standard balanced split used in this run])

From an image-processing perspective, CIFAR-10 is demanding because the objects are semantically rich but spatially tiny. A $32 times 32$ image contains only 1024 spatial samples per channel, so object boundaries, texture, background context, and color cues are heavily compressed. Good performance therefore depends on filters that can capture local edge/color patterns while preserving enough spatial layout for object-level discrimination.

= Input representation and preprocessing

The implementation indexes images from class directories, decodes PNG/JPEG images with the `stb_image` open source library, forces three RGB channels, resizes each image to $32 times 32$ using bilinear interpolation and normalizes pixel values from integer byte values to floating point values in the range [0, 1]. The output tensor layout is NCHW:

- *N*: batch size,
- *C*: channel count (3 for RGB),
- *H*: height (32 pixels),
- *W*: width (32 pixels).

The preprocessing can be summarized as:

```
file image -> RGB decode -> bilinear resize to 32x32 -> divide by 255 -> NCHW tensor
```

The source constants fix the image shape and class count:

```c
// dataset.h
#define NUM_CLASSES    10
#define IMAGE_WIDTH    32
#define IMAGE_HEIGHT   32
#define IMAGE_CHANNELS 3
```

The dataset loader confirms the class-directory assumption and the CIFAR-10 class order used by the program:

```c
// dataset.c
const char *CIFAR_CLASS_NAMES[NUM_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Labels are subfolders in dataset directory, each named after the class
for (int label = 0; label < NUM_CLASSES; label++) {
    if (!load_class_directory(dataset, root_dir,CIFAR_CLASS_NAMES[label], label)) 
        return 0;
}
```

The actual pixel preprocessing is implemented as bilinear interpolation followed by NCHW storage and normalization:

```c
// dataset.c, shortened excerpt
// Bilinear resizing and normalization to RGB [0,1] in NCHW format
static void resize_bilinear(const unsigned char *src,
                            int src_w, int src_h, float *dst) {
    const int dst_w = IMAGE_WIDTH;
    const int dst_h = IMAGE_HEIGHT;
    const int channels = IMAGE_CHANNELS;
    const int dst_image_size = dst_w * dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        float src_y = ((float)dy + 0.5f) *
                      ((float)src_h / (float)dst_h) - 0.5f;
        int y0 = (int)floorf(src_y);
        float wy = src_y - (float)y0;
        if (y0 < 0) { y0 = 0; wy = 0.0f; }
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;

        for (int dx = 0; dx < dst_w; dx++) {
            float src_x = ((float)dx + 0.5f) *
                          ((float)src_w / (float)dst_w) - 0.5f;
            int x0 = (int)floorf(src_x);
            float wx = src_x - (float)x0;
            if (x0 < 0) { x0 = 0; wx = 0.0f; }
            int x1 = x0 + 1;
            if (x1 >= src_w) x1 = src_w - 1;

            for (int c = 0; c < channels; c++) {
                int idx00 = (y0 * src_w + x0) * channels + c;
                int idx01 = (y0 * src_w + x1) * channels + c;
                int idx10 = (y1 * src_w + x0) * channels + c;
                int idx11 = (y1 * src_w + x1) * channels + c;

                float p00 = (float)src[idx00];
                float p01 = (float)src[idx01];
                float p10 = (float)src[idx10];
                float p11 = (float)src[idx11];

                float top = p00 * (1.0f - wx) + p01 * wx;
                float bot = p10 * (1.0f - wx) + p11 * wx;
                float val = top * (1.0f - wy) + bot * wy;

                int dst_idx = c * dst_image_size + dy * dst_w + dx;
                dst[dst_idx] = val / 255.0f;
            }
        }
    }
}
```

This is a minimal and reproducible preprocessing pipeline. It is adequate for verifying a custom CUDA network, but it isn't optimal in terms of performance because it lacks common data augmentation and normalization techniques. CIFAR-10 is already $32 times 32$ in its native form, so resizing is useful only if the data have been converted into image files with uncertain dimensions. The implementation also does not apply per-channel mean/std normalization, random crop, horizontal flip, color jitter, or cutout-style augmentation. These omissions matter because CIFAR-10 classification benefits from invariance to small translations, mirror transformations, lighting changes, and background clutter.

= Network architecture

The network is a compact CNN with a single learned convolutional feature extraction stage followed by a fully connected classifier. It can be described as a shallow LeNet-style architecture:

#figure(table(
  columns: (auto, auto, auto, 4.5cm),
  inset: 5pt,
  align: (left, left, left, left),
  [*Stage*],
  [*Operation*],
  [*Output tensor*],
  [*Image-processing role*],

  [Input],
  [RGB image],
  [$3 times 32 times 32$],
  [Three color channels at low spatial resolution.],

  [Convolution],
  [32 kernels, $5 times 5$, stride 1, padding 2],
  [$32 times 32 times 32$],
  [Learns local spatial/color filters while preserving image size.],

  [Activation],
  [ReLU],
  [$32 times 32 times 32$],
  [Keeps positive filter responses and introduces nonlinearity.],

  [Pooling],
  [2×2 max pool, stride 2],
  [$32 times 16 times 16$],
  [Downsamples feature maps and adds local translation robustness.],

  [Flatten],
  [Vectorization],
  [$8192$],
  [Converts spatial feature maps to classifier input.],

  [Classifier],
  [Fully connected + bias],
  [$10$ logits],
  [Maps extracted features to class scores.],

  [Output],
  [Softmax + argmax],
  [$10$ probabilities + prediction],
  [Produces normalized class confidence and predicted label.],
))

The architecture constants are compiled from the header file:

```c
// kernels.h
#define KERNEL_H 5
#define KERNEL_W 5
#define KERNEL_COUNT 32

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
#define EPOCHS 10
#define FIXED_SEED 123
```
The following table summarizes the parameter count and role of each learnable component. The convolutional kernels are the only parameters in the feature extractor, while the fully connected weights and bias are the only parameters in the classifier:

#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 5pt,
  align: (left, right, right, left),
  [*Parameter group*], [*Shape*], [*Count*], [*Comment*],
  [Convolution kernels],
  [$32 times 3 times 5 times 5$],
  [2,400],
  [Local color-spatial filters.],

  [Fully connected weights],
  [$8192 times 10$],
  [81,920],
  [Classifier matrix after flattening.],

  [Fully connected bias], [$10$], [$10$], [One bias per class.],
  [Total implemented],
  [],
  [84,330],
  [97.1% of parameters are in the FC weights.],
))

It is worth noting that the flatten size of 8192 is large relative to the network depth. This means that most parameters and most measured forward time reside in the classifier, which is a limitation of the current architecture. Also note that the convolutional kernels are the only learnable parameters in the feature extractor, so the network's ability to extract useful features is limited by the number of kernels and their size.

= CUDA implementation and training loop

The forward pass is decomposed into five timed GPU stages: convolution, ReLU, max pooling, fully connected plus bias, and softmax plus prediction. The convolution kernel uses shared memory tiling. The classifier uses a shared-memory matrix multiplication kernel to compute logits from the flattened pooled tensor.

The main training loop exposes the host-side learning protocol: shuffle the training index, load and preprocess a batch, transfer it to the GPU, run forward propagation, run backpropagation, compute loss, and accumulate accuracy.

```c
// main.cu, shortened excerpt
const float learningRate = 0.001f;
const float lambda = 1e-4f;

for (int epoch = 0; epoch < EPOCHS; epoch++) {
    shuffle_dataset(&train);

    for (int batch = 0; batch < trainBatches; batch++) {
        int startIndex = batch * BATCH_SIZE;
        int loaded = load_batch(&train, startIndex,
                                BATCH_SIZE, h_input, h_labels);
        if (loaded != BATCH_SIZE) continue;

        cudaMemcpy(d_input, h_input,
                   (size_t)inputElements * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, h_labels,
                   (size_t)BATCH_SIZE * sizeof(int),
                   cudaMemcpyHostToDevice);

        LeNet_forward(d_input, d_labels, cnn,
                      collectTiming ? forwardTiming : NULL);
        LeNet_backward(d_input, d_labels, cnn,
                       learningRate, lambda, NULL);

        float batchLoss = compute_batch_loss(cnn->d_softmax_output,
                                             d_labels, d_loss,
                                             h_loss, BATCH_SIZE);
    }
}
```

The forward pass maps directly to the architecture table:

```c
// lenet.cu, shortened forward-pass excerpt
convolutionSharedKernel<<<convGridDim, convBlockDim, sharedMemSize>>>(
    d_input, cnn->d_kernels, cnn->d_conv_output,
    BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, KERNEL_SIZE,
    KERNEL_COUNT, OUTPUT_SIZE, PADDING, STRIDE);

cudaMemcpy(cnn->d_activation, cnn->d_conv_output,
           convTotalElements * sizeof(float), cudaMemcpyDeviceToDevice);
reluActivationKernel<<<reluGridSize, blockSize>>>(
    cnn->d_activation, convTotalElements);

maxPoolingKernel<<<poolGridDim, poolBlockDim>>>(
    cnn->d_activation, cnn->d_pooling_output,
    BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
    POOL_SIZE, POOL_OUTPUT_SIZE, POOL_STRIDE);

matrixMultiplySharedKernel<<<mmGridDim, mmBlockDim>>>(
    cnn->d_pooling_output, cnn->d_fc_weights, cnn->d_logits,
    BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);
addBiasKernel<<<biasGridDim, biasBlockDim>>>(
    cnn->d_logits, cnn->d_fc_bias, BATCH_SIZE, NUM_CLASSES);

softmaxKernel<<<predGridSize, predBlockSize>>>(
    cnn->d_logits, cnn->d_softmax_output, BATCH_SIZE, NUM_CLASSES);
getPredictionsKernel<<<predGridSize, predBlockSize>>>(
    cnn->d_softmax_output, cnn->d_predictions, BATCH_SIZE, NUM_CLASSES);
```

The backward pass follows the usual cross-entropy/softmax training path:

#figure(table(
  columns: (auto, auto),
  inset: 5pt,
  align: (left, left),
  [*Backward stage*], [*Purpose*],
  [Softmax cross-entropy gradient],
  [Computes `(softmax - one_hot(label)) / batch_size` for the logits.],

  [FC gradients],
  [Computes fully connected weight, bias, and input gradients.],

  [Max-pooling backward],
  [Routes gradients only to maximum locations selected by the forward pooling window.],

  [ReLU backward],
  [Suppresses gradients where pre-ReLU convolution responses were non-positive.],

  [Convolution weight gradient],
  [Reduces gradients over batch, spatial position, and input channel dimensions.],

  [L2 regularization and SGD],
  [Adds ridge-style weight decay and applies SGD updates.],
))

The source-level implementation of those stages is visible in the following backpropagation excerpt:

```c
// lenet.cu, shortened backward-pass excerpt
softmaxCrossEntropyKernel<<<logitsGrid, blockSize>>>(
    cnn->d_softmax_output, d_labels, cnn->d_d_logits,
    BATCH_SIZE, NUM_CLASSES);

fcWeightGradientKernel<<<fcWeightGrid, fcWeightBlock>>>(
    cnn->d_pooling_output, cnn->d_d_logits, cnn->d_d_fc_weights,
    BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);
fcBiasGradientKernel<<<biasGrid, blockSize>>>(
    cnn->d_d_logits, cnn->d_d_fc_bias, BATCH_SIZE, NUM_CLASSES);
fcInputGradientKernel<<<fcInputGrid, fcInputBlock>>>(
    cnn->d_d_logits, cnn->d_fc_weights, cnn->d_d_pooling_output,
    BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);

cudaMemset(cnn->d_d_activation, 0,
           convTotalElements * sizeof(float));
maxPoolingBackwardKernel<<<poolGridDim, poolBlockDim>>>(
    cnn->d_activation, cnn->d_d_pooling_output, cnn->d_d_activation,
    BATCH_SIZE, KERNEL_COUNT, OUTPUT_SIZE,
    POOL_SIZE, POOL_OUTPUT_SIZE, POOL_STRIDE);

reluBackwardKernel<<<reluGrid, blockSize>>>(
    cnn->d_conv_output, cnn->d_d_activation,
    cnn->d_d_conv_output, convTotalElements);

conv2WeightGradientReduceKernel<<<convWGrid, blockSize,
                                  blockSize * sizeof(float)>>>(
    d_input, cnn->d_d_conv_output, cnn->d_d_kernels,
    BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE,
    KERNEL_COUNT, KERNEL_SIZE, KERNEL_SIZE,
    OUTPUT_SIZE, OUTPUT_SIZE, PADDING, PADDING, STRIDE, STRIDE);

ridgeL2GradientKernel<<<fcUpdateGrid, blockSize>>>(
    cnn->d_d_fc_weights, cnn->d_fc_weights, fcWeights, lambda);
sgdUpdateKernel<<<fcUpdateGrid, blockSize>>>(
    cnn->d_fc_weights, cnn->d_d_fc_weights, learningRate, fcWeights);
```

The run used a fixed seed of 123, batch size 16, learning rate 0.001, L2 coefficient 1e-4, and ten epochs. Each epoch used 3125 training batches and 625 test batches, matching exactly 50,000 training and 10,000 test images with batch size 16.

= Results after ten epochs

The epoch-level results show steady optimization over all ten epochs.

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (center, right, right, right, right),
  [*Epoch*],
  [*Train loss*],
  [*Train acc.*],
  [*Test loss*],
  [*Test acc.*],

  [1], [2.0985], [24.65%], [1.9355], [32.85%],
  [2], [1.8700], [34.99%], [1.8339], [35.11%],
  [3], [1.7682], [38.83%], [1.7208], [41.26%],
  [4], [1.6844], [41.88%], [1.6530], [43.03%],
  [5], [1.6107], [44.31%], [1.5793], [44.98%],
  [6], [1.5461], [46.71%], [1.5197], [47.51%],
  [7], [1.4916], [48.59%], [1.4678], [48.87%],
  [8], [1.4466], [50.07%], [1.4307], [50.08%],
  [9], [1.4096], [51.30%], [1.4026], [50.73%],
  [10], [1.3812], [52.26%], [1.3848], [51.75%],
))

The curves below visualize the same data.

#figure(grid(
  columns: (1fr, 1fr),
  figure(
    image("loss_curve.png", width: 100%),
    caption: [Cross-entropy loss over epochs.],
  ),
  figure(
    image("accuracy_curve.png", width: 100%),
    caption: [Classification accuracy over epochs.],
  ),
))

The most important observations are:

- Training and test accuracy improve monotonically at the epoch-summary level, ending at 52.26% and 51.75% respectively.
- The final train-test accuracy gap is only 0.51 percentage points, so the network is not strongly overfitting.
- The final test accuracy is far above the 10% random-choice baseline, but below what deeper CIFAR-10 CNNs can achieve and below the Python/Keras LeNet baseline discussed later, which reaches 58.89% in the provided run.
- The convergence curve has not saturated completely after ten epochs; later epochs, a learning-rate schedule, or a stronger optimizer could plausibly improve the result.

= Timing analysis

The kernel-level timing was measured after every ten batches and at the last batch of each phase. In total, the parsed timing set contains 3760 forward timing lines and 3130 training backward timing lines. The average measured forward pass on training batches is 8.164 ms; the average measured forward pass on test batches is 8.037 ms. The average measured backward pass on logged training batches is 2.193 ms. Thus, the measured training compute path is approximately 10.36 ms per logged batch, excluding data loading and host-to-device transfer overheads.

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right, right),
  [*Forward component*],
  [*Train mean ms*],
  [*Train share*],
  [*Test mean ms*],
  [*Test share*],

  [Conv], [0.722], [8.84%], [0.727], [9.05%],
  [ReLU], [0.012], [0.14%], [0.013], [0.16%],
  [Max pool], [0.023], [0.28%], [0.024], [0.29%],
  [FC + bias], [7.372], [90.30%], [7.239], [90.08%],
  [Softmax + prediction], [0.035], [0.43%], [0.034], [0.43%],
))

#grid(
  columns: (1fr, 1fr),
  figure(
    image("timing_breakdown.png", width: 100%),
    caption: [Mean training forward-pass timing by component.],
  ),
  figure(
    image("forward_time_by_epoch.png", width: 100%),
    caption: [Mean logged forward time per epoch.],
  ),
)

Within the measured GPU forward pass, `FC+Bias` accounts for about 90% of measured forward time, even though the convolution has more theoretical multiply-add work. This indicates that the measured GPU bottleneck is implementation-level rather than purely arithmetic. Likely contributors include memory access patterns, small matrix dimensions, launch overhead, and the custom matrix multiplication kernel. The fully convolutional comparison in Section 10 tests this diagnosis directly by replacing the flatten-plus-FC forward head with a mathematically equivalent valid convolution. This statement should not be confused with the end-to-end program bottleneck: the timing lines measure CUDA events around GPU stages and exclude file-system I/O, PNG/JPEG decompression through `stbi_load`, CPU bilinear preprocessing, and host-to-device transfer. At system level, the pure CUDA implementation is bottlenecked primarily by the image-file data path. Replacing per-batch image-file reads with a binary or cached tensor loader would improve the full training program.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 4pt,
    align: (center, right, right),
    [*Epoch*], [*Train forward ms*], [*Test forward ms*],
    [1], [7.781], [7.816],
    [2], [7.878], [7.913],
    [3], [7.966], [7.988],
    [4], [8.070], [8.086],
    [5], [8.116], [8.093],
    [6], [8.246], [8.064],
    [7], [8.360], [8.067],
    [8], [8.360], [8.104],
    [9], [8.409], [8.115],
    [10], [8.459], [8.121],
  ),
  caption: [Mean forward time per epoch],
)

The gradual increase in logged training forward time across epochs is small but visible. It is not caused by changing tensor shapes, so it is more likely due to runtime variability, timing overhead, GPU state, or memory/cache effects than the mathematical model itself.


= Comparison with the Python/Keras LeNet baseline

A second experiment was provided in `lenet.py`. It is useful as a reference implementation, but it is not a direct comparison of two identical nets. The Python version uses Keras, the standard CIFAR-10 loader, a deeper LeNet-style architecture, and the Adam optimizer. The pure CUDA version uses a custom C/CUDA data pipeline, one convolutional feature-extraction layer, a direct flatten-to-softmax classifier, and SGD with L2 regularization.

== Data path and preprocessing

The most important implementation difference is the data path. The Python script loads CIFAR-10 once through the Keras dataset API and then normalizes already materialized arrays:

```python
# lenet.py, data path
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
```

This means that, during training, Keras iterates over in-memory tensors rather than repeatedly opening one image file per sample. By contrast, the pure CUDA program stores paths to PNG/JPEG files and loads the actual pixels inside `load_batch`:

```c
// dataset.c, batch-time data loading path
for (int n = 0; n < batch_size; n++) {
    int sample_index = start_index + n;
    if (sample_index >= dataset->count) break;

    int src_w = 0, src_h = 0, src_channels = 0;
    unsigned char *src = stbi_load(
        dataset->samples[sample_index].path,
        &src_w, &src_h, &src_channels, IMAGE_CHANNELS
    );

    float *dst = images + (size_t)loaded * output_image_size;
    resize_bilinear(src, src_w, src_h, dst);

    labels[loaded] = dataset->samples[sample_index].label;
    stbi_image_free(src);
    loaded++;
}
```

This is the reason why the end-to-end CUDA implementation is bottlenecked by file reading and image decoding. The kernel timing table measures CUDA events around GPU computation; it does not include the CPU-side cost of opening files, decoding PNG/JPEG data, resizing, normalizing, filling host buffers, or copying a batch to the GPU. With 50,000 training images, 10,000 test images, and testing after each epoch, the program performs approximately $10 times (50000 + 10000) = 600000$ batch-time image decodes, in addition to the initial dataset-indexing pass that verifies decodability. This overhead is absent from the Keras path because the CIFAR-10 arrays are loaded up front.

== Architecture and optimizer differences

The Python model is closer to the classical LeNet pattern: two convolution/pooling stages followed by two hidden dense layers and an output softmax. The pure CUDA network is shallower in the convolutional part but has a much larger flatten vector because it preserves 32 feature maps at $16 times 16$ before classification.

```python
# lenet.py, model structure
model.add(Conv2D(6, kernel_size=(5, 5), activation="relu",
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation="relu"))
model.add(Dense(84, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
```

#figure(table(
  columns: (auto, 5.6cm, 5.6cm),
  inset: 4pt,
  align: (left, left, left),
  [*Aspect*], [*Python/Keras LeNet*], [*Pure CUDA implementation*],

  [Input format],
  [CIFAR-10 arrays loaded by `datasets.cifar10.load_data()`.],
  [Image-folder tree: `dataset/train/<class-name>` and `dataset/test/<class-name>`.],

  [Preprocessing],
  [Divide tensor values by 255.0; no explicit resize because CIFAR-10 is already $32 times 32$.],
  [Decode PNG/JPEG with `stbi_load`, force RGB, bilinear resize to $32 times 32$, divide by 255, store NCHW.],

  [Feature extractor],
  [`Conv2D(6, 5×5, valid) -> pool -> Conv2D(16, 5×5, valid) -> pool`.],
  [`Conv(32, 5×5, stride 1, padding 2) -> ReLU -> 2×2 max pool`.],

  [Classifier],
  [`Flatten(400) -> Dense(120) -> Dense(84) -> Dense(10)`.],
  [`Flatten(8192) -> FC(10)`.],

  [Parameters],
  [62,006 trainable parameters.],
  [84,330 implemented parameters.],

  [Optimizer],
  [Adam with categorical cross-entropy.],
  [Plain SGD with learning rate 0.001 and L2 coefficient $10^(-4)$.],

  [Final train result],
  [Loss 0.8658, accuracy 69.37%.],
  [Loss 1.3812, accuracy 52.26%.],

  [Final test result],
  [Loss 1.2056, accuracy 58.89%.],
  [Loss 1.3848, accuracy 51.75%.],

  [Main bottleneck],
  [Mostly hidden inside Keras/TensorFlow kernels and in-memory tensor iteration.],
  [At system level: repeated file open/decode/resize from individual images; within measured GPU forward time: FC+Bias.],
), caption: [Comparison between the Python/Keras LeNet and the pure CUDA implementation])

The Python run improves training accuracy from 41.05% after epoch 1 to 69.37% after epoch 10, and reports a final test accuracy of 58.89%. The pure CUDA run improves test accuracy from 32.85% after epoch 1 to 51.75% after epoch 10. The Python model therefore generalizes about 7.14 percentage points better in this experiment, but the difference should be interpreted as a combination of architecture, optimizer, library kernels, and data pipeline rather than as a direct language comparison.

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (center, right, right, right, right),
  [*Epoch*], [*Python train loss*], [*Python train acc.*], [*CUDA train acc.*], [*CUDA test acc.*],
  [1], [1.6034], [41.05%], [24.65%], [32.85%],
  [2], [1.3124], [52.84%], [34.99%], [35.11%],
  [3], [1.2001], [57.40%], [38.83%], [41.26%],
  [4], [1.1204], [60.10%], [41.88%], [43.03%],
  [5], [1.0608], [62.33%], [44.31%], [44.98%],
  [6], [1.0101], [64.45%], [46.71%], [47.51%],
  [7], [0.9665], [65.91%], [48.59%], [48.87%],
  [8], [0.9306], [66.96%], [50.07%], [50.08%],
  [9], [0.8930], [68.50%], [51.30%], [50.73%],
  [10], [0.8658], [69.37%], [52.26%], [51.75%],
), caption: [Training progression of the Python LeNet compared with the CUDA run])

== Interpretation of the comparison

The Python/Keras implementation is stronger for classification because it has two convolutional stages and two hidden dense layers, so it can build a small hierarchy of features before classification. It also uses Adam, which adapts the update scale for each parameter and typically converges faster than plain SGD at the beginning of training. The pure CUDA implementation is more useful as a transparent systems baseline: every GPU kernel, tensor layout, memory copy, and backward-pass stage is visible and controllable.

For image-processing experimentation, the CUDA version should not be judged only by its current end-to-end speed. Its GPU kernels are timed separately and are reasonably stable, but the input pipeline converts a dataset that was originally distributed in compact binary batches into thousands of small image files. This is a poor layout for high-throughput training. The best correction is to add a binary CIFAR-10 reader or a one-time preprocessing step that writes a contiguous tensor cache, then train from that cache. After that change, the remaining bottlenecks would be the custom FC matrix multiplication and the shallow model design rather than file I/O.

= Image-processing interpretation of the learned pipeline

== Convolution as learned spatial filtering

The first layer performs 32 learned 5×5 convolutions over RGB input. In classical image processing, such filters can be interpreted as local spatial operators responding to color contrast, edges, corners, and small texture primitives. The padding value of 2 preserves the $32 times 32$ spatial grid, which is useful because CIFAR-10 objects already occupy few pixels.

The CUDA convolution explicitly computes each output value as a local weighted sum over input channel, kernel row, and kernel column. The indexing below also shows the NCHW layout used by the implementation.

```c
// kernels.cu, shortened convolution inner loop
for (int c = 0; c < inputChannels; c++) {
    for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
            int shared_y = ty * stride + ky;
            int shared_x = tx * stride + kx;

            float in_val = sharedInput[
                c * tileSizeWithPadding * tileSizeWithPadding +
                shared_y * tileSizeWithPadding + shared_x];

            float kernel_val = kernels[
                k * inputChannels * kernelSize * kernelSize +
                c * kernelSize * kernelSize + ky * kernelSize + kx];

            sum += in_val * kernel_val;
        }
    }
}

output[b * kernelCount * outputSize * outputSize +
       k * outputSize * outputSize + out_y * outputSize + out_x] = sum;
```

A limitation is that one convolutional layer cannot build a hierarchy of increasingly abstract image features. A deeper design would allow early filters to detect oriented edges and color blobs, middle filters to combine them into parts and textures, and later filters to assemble object-level cues. The current model jumps from one low-level filter bank directly to a large classifier.

== ReLU as non-linear feature selection

ReLU keeps positive filter responses and suppresses negative ones. This makes the representation sparse and non-linear. In image-processing terms, it acts like a half-wave rectifier applied to filter responses, retaining strong evidence for the learned local patterns.

```c
// kernels.cu
__global__ void reluActivationKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = fmaxf(0.0f, data[idx]);
}
```

== Max pooling as local invariance and downsampling

The 2×2 max-pooling stage halves the spatial size from $32 times 32$ to 16×16. This reduces feature-map storage by a factor of four and makes the classifier less sensitive to one-pixel shifts. The cost is loss of precise spatial detail. On CIFAR-10, this is a reasonable tradeoff because object category is more important than exact pixel alignment, but aggressive pooling in a very shallow network can also discard useful localization information.

```c
// kernels.cu, shortened max-pooling excerpt
float maxVal = -FLT_MAX;

for (int dy = 0; dy < poolSize; dy++) {
    for (int dx = 0; dx < poolSize; dx++) {
        int in_y = in_y_base + dy;
        int in_x = in_x_base + dx;

        if (in_y < inputSize && in_x < inputSize) {
            float value = input[b * channels * inputSize * inputSize +
                                c * inputSize * inputSize +
                                in_y * inputSize + in_x];
            maxVal = fmaxf(maxVal, value);
        }
    }
}

output[b * channels * outputSize * outputSize +
       c * outputSize * outputSize + out_y * outputSize + out_x] = maxVal;
```

== Fully connected layer as global template classifier

After pooling, the 32 feature maps are flattened to 8192 values. The classifier then learns ten global templates over this flattened feature vector. This makes the output decision depend on absolute feature-map positions. That can help when object placement is consistent, but it is weaker when objects translate, rotate, or appear against cluttered backgrounds.

The source uses a tiled matrix multiplication kernel for this stage, followed by an explicit bias-add kernel:

```c
// kernels.cu, shortened matrix multiplication excerpt
__shared__ float sA[TILE_SIZE][TILE_SIZE];
__shared__ float sB[TILE_SIZE][TILE_SIZE];

for (int tile = 0; tile < numTiles; tile++) {
    int aCol = tile * TILE_SIZE + threadIdx.x;
    int bRow = tile * TILE_SIZE + threadIdx.y;

    sA[threadIdx.y][threadIdx.x] =
        (row < A_rows && aCol < A_cols) ? A[row * A_cols + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] =
        (bRow < A_cols && col < B_cols) ? B[bRow * B_cols + col] : 0.0f;

    __syncthreads();
    for (int k = 0; k < TILE_SIZE; k++)
        sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
}

if (row < A_rows && col < B_cols)
    C[row * B_cols + col] = sum;
```

== Softmax and prediction

The final image-classification step converts logits to probabilities with a max-shifted softmax and chooses the maximum-probability class.

```c
// kernels.cu, shortened softmax excerpt
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
```


= Comparison with the fully convolutional equivalent

The original CUDA classifier flattens the pooled tensor and multiplies it by a dense weight matrix. The fully convolutional version uses the same learned parameters but views the dense classifier as a convolution whose kernel spans the entire pooled spatial field. For a pooled tensor with shape $32 times 16 times 16$, the flattened vector has $8192$ elements and the original classifier weight matrix has shape $8192 times 10$. The FCN version reshapes those weights into ten filters of shape $32 times 16 times 16$ and applies a valid convolution, producing $10$ class logits at a $1 times 1$ output location.

This is not a new higher-capacity model. It is an implementation-level equivalence: `Flatten(pool) -> FC(8192, 10) + bias` becomes `Conv2D(10 filters, kernel=16x16 over 32 channels) + bias`. The trainable parameter count remains the same: 81,920 head weights plus ten biases. The main benefit is that the classifier is expressed with convolutional indexing and a parallel reduction over the feature dimension instead of the previous custom small-matrix multiplication path.

== Source-level implementation

The FCN support code is additive: `main_fcn.cu` leaves the original `main.cu` path unchanged, wraps the same `LeNet` object, and replaces only the forward classification head. The weight conversion in `kernels_fcn.cu` maps a dense flatten/class index to a convolution filter index:

```c
// kernels_fcn.cu, FC weight layout -> Conv2D filter layout
int flattenIndex = (c * kernelH + ky) * kernelW + kx;

// Conv2D layout: [class, channel, ky, kx]
// FC layout:     [flattenIndex, class]
convWeights[idx] = fcWeights[flattenIndex * numClasses + cls];
```

The FC-as-convolution kernel then computes one output logit per `(batch, class)` block. Threads accumulate partial sums over the $32 times 16 times 16$ input feature field and reduce them in shared memory:

```c
// kernels_fcn.cu, shortened FC-as-Conv2D valid convolution
for (int f = tid; f < features; f += blockDim.x) {
    int tmp = f;
    int x = tmp % inputW;
    tmp /= inputW;
    int y = tmp % inputH;
    tmp /= inputH;
    int c = tmp;

    int inputIdx = ((b * inChannels + c) * inputH + y) * inputW + x;
    int weightIdx = ((cls * inChannels + c) * inputH + y) * inputW + x;

    localSum += input[inputIdx] * convWeights[weightIdx];
}

sharedSum[tid] = localSum;
__syncthreads();

for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset)
        sharedSum[tid] += sharedSum[tid + offset];
    __syncthreads();
}

if (tid == 0)
    logits[b * numClasses + cls] = sharedSum[0] + bias[cls];
```

In `lenet_fcn.cu`, `LeNetFCN_forward` synchronizes the reshaped convolution filters from the current dense FC weights before running the forward pass. The training executable still calls the original backward path, because the logits are equivalent and the existing gradient code already updates the dense FC weights:

```c
// main_fcn.cu, shortened training step
LeNetFCN_forward(d_input, d_labels, fcn,
                 collectTiming ? forwardTiming : NULL);

// Original backward path remains unchanged.
LeNet_backward(d_input, d_labels, cnn,
               learningRate, lambda, NULL);
```

== Equivalence check

The standalone equivalence test compares the original FC forward path and the FCN forward path on the same batch and weights. The result shows that the numerical differences are at single-precision roundoff scale and do not change predicted classes:

#figure(table(
  columns: (auto, auto),
  inset: 5pt,
  align: (left, right),
  [*Check*], [*Measured result*],
  [`Max |logits_fc - logits_fcn|`], [$1.13248825 times 10^(-6)$],
  [`Max |softmax_fc - softmax_fcn|`], [$8.94069672 times 10^(-8)$],
  [Prediction mismatches], [0/16],
), caption: [FC vs FCN equivalence check])

The small logit and softmax differences are expected because the FC and FCN paths sum the same floating-point products in different orders. For classification, the decisive result is that all 16 predictions agree.

== Accuracy and loss comparison

Because the FCN version is an equivalent forward reparameterization of the same classifier, its learning curve should match the original run apart from tiny differences introduced by floating-point reduction order. That is what the measurements show. The final losses match to four decimal places, and the final accuracies differ only by hundredths of a percentage point.

#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right),
  [*Metric*], [*Original FC*], [*FC-as-Conv2D*], [*Difference*],
  [Final train loss], [1.3812], [1.3812], [0.0000],
  [Final train accuracy], [52.26%], [52.27%], [+0.01 pp],
  [Final test loss], [1.3848], [1.3848], [0.0000],
  [Final test accuracy], [51.75%], [51.73%], [-0.02 pp],
  [Used training batches], [3125/3125], [3125/3125], [same],
  [Used test batches], [625/625], [625/625], [same],
), caption: [Classification results of the original FC head and its FCN-equivalent head])

The accuracy comparison confirms that the FCN run should be interpreted as a systems and implementation comparison rather than as a representation-learning improvement. It preserves the same shallow feature extractor, same parameter count, same training data, same optimizer, and same fixed seed.

== Timing comparison

The timing difference is large because the original classifier dominated the measured forward pass. In the original forward timing, `FC+Bias` used 7.372 ms on average during training logging and represented 90.30% of the forward pass. In the FCN version, the corresponding `FC-as-Conv2D+Bias` stage uses 0.111 ms and represents 11.18% of the forward pass. The mean logged training forward pass drops from 8.164 ms to 0.989 ms, an 8.25× speedup. The mean logged test forward pass drops from 8.037 ms to 0.923 ms, an 8.71× speedup.

#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right),
  [*Timing metric*], [*Original FC*], [*FC-as-Conv2D*], [*Speedup*],
  [Mean train forward], [8.164 ms], [0.989 ms], [8.25×],
  [Mean test forward], [8.037 ms], [0.923 ms], [8.71×],
  [Mean train classifier head], [7.372 ms], [0.111 ms], [66.65×],
  [Mean test classifier head], [7.239 ms], [0.103 ms], [70.19×],
  [Mean logged train forward + backward], [10.357 ms], [3.351 ms], [3.09×],
), caption: [Forward timing comparison between the original FC path and the FCN-equivalent path])

#figure(
  image("fcn_timing_comparison.png", width: 100%),
  caption: [Mean CUDA event timing for the original FC head and the FC-as-Conv2D head.],
)

The FCN component breakdown also changes the interpretation of the measured GPU bottleneck. After the head rewrite, convolution becomes the largest forward component instead of the classifier head:

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right, right),
  [*Forward component*],
  [*FCN train mean ms*],
  [*FCN train share*],
  [*FCN test mean ms*],
  [*FCN test share*],
  [Conv], [0.802], [81.08%], [0.748], [81.05%],
  [ReLU], [0.012], [1.22%], [0.012], [1.25%],
  [Max pool], [0.025], [2.55%], [0.024], [2.56%],
  [FC-as-Conv2D + bias], [0.111], [11.18%], [0.103], [11.18%],
  [Softmax + prediction], [0.039], [3.99%], [0.037], [3.96%],
), caption: [FCN-equivalent forward-pass timing breakdown])

The backward pass is unchanged because the FCN forward path is mathematically equivalent to the original FC path. The backward pass still uses the original dense-FC gradient code, so the mean logged training backward time remains 2.193 ms.

== Interpretation

The fully convolutional comparison supports two conclusions. First, the mathematical classifier can be written either as a dense layer over a flattened vector or as a valid convolution over the whole pooled feature map; for fixed $32 times 32$ CIFAR-10 inputs these two forms are functionally equivalent. Second, the original timing bottleneck was not inherent to the number of multiply-adds in the classifier, but to the particular custom FC implementation used for a very small $16 times 8192$ by $8192 times 10$ multiplication. The FCN head exposes more parallelism per `(batch, class)` output and avoids the original small-GEMM overhead, so it removes the measured forward bottleneck while preserving the image-processing pipeline and classification behavior.

The current FCN executable still uses fixed CIFAR-10 input geometry and a $16 times 16$ classifier kernel, so it should not be described as a fully general dense-prediction network. However, it is a useful bridge between the current classifier and more standard convolutional designs: once the dense head is expressed as a convolution, later experiments can replace the global $16 times 16$ kernel with smaller convolutional blocks plus global average pooling, or slide the classifier over larger feature maps for spatial class-score maps.


= Measurements on NVIDIA Jetson

The original FC executable and the FC-as-Conv2D executable were also run on an NVIDIA Jetson using the same CIFAR-10 split, batch size 16, ten epochs, and the same network tensor sizes reported earlier. The Jetson given by the University has several Intel Xeon on board.

== Jetson accuracy and loss

The Jetson learning curves are effectively identical for the original dense classifier head and the FC-as-Conv2D forward head. The final epoch reaches 52.32% training accuracy and 52.41% test accuracy. The FCN version preserves the same losses and accuracies to the precision printed by the logger.

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (center, right, right, right, right),
  [*Epoch*],
  [*Original FC test loss*],
  [*Original FC test acc.*],
  [*FC-as-Conv2D test loss*],
  [*FC-as-Conv2D test acc.*],
  [1], [1.9353], [32.72%], [1.9353], [32.72%],
  [2], [1.8165], [36.52%], [1.8165], [36.52%],
  [3], [1.7241], [40.53%], [1.7241], [40.53%],
  [4], [1.6515], [42.66%], [1.6515], [42.66%],
  [5], [1.5780], [45.07%], [1.5780], [45.07%],
  [6], [1.5132], [47.68%], [1.5132], [47.68%],
  [7], [1.4680], [48.84%], [1.4680], [48.84%],
  [8], [1.4333], [49.91%], [1.4333], [49.91%],
  [9], [1.4009], [51.03%], [1.4009], [51.03%],
  [10], [1.3795], [52.41%], [1.3795], [52.41%],
), caption: [Jetson test-set learning curve for the original FC head and the FC-as-Conv2D head])

#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right),
  [*Final Jetson metric*], [*Original FC*], [*FC-as-Conv2D*], [*Difference*],
  [Train loss], [1.3785], [1.3785], [0.0000],
  [Train accuracy], [52.32%], [52.32%], [0.00 pp],
  [Test loss], [1.3795], [1.3795], [0.0000],
  [Test accuracy], [52.41%], [52.41%], [0.00 pp],
  [Used training batches], [3125/3125], [3125/3125], [same],
  [Used test batches], [625/625], [625/625], [same],
), caption: [Final Jetson accuracy and loss])

These results confirm that the FC-as-Conv2D path is an implementation change rather than a representational change. On Jetson, as in the first timing run, the classifier-head rewrite does not improve accuracy; it preserves the model behavior while changing the timing profile.

== Jetson timing

The Jetson timing logs contain 3760 forward timing lines and 3130 training backward timing lines for each executable. The original dense-head run has a mean logged training forward time of 0.638 ms and a mean logged test forward time of 0.638 ms. The FC-as-Conv2D run has a mean logged training forward time of 0.193 ms and a mean logged test forward time of 0.196 ms. Because the backward path is intentionally unchanged, the mean backward time remains approximately 0.722 ms in both runs.

#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right),
  [*Jetson timing metric*], [*Original FC*], [*FC-as-Conv2D*], [*Speedup*],
  [Mean train forward], [0.638 ms], [0.193 ms], [3.31×],
  [Mean test forward], [0.638 ms], [0.196 ms], [3.26×],
  [Mean train classifier head], [0.470 ms], [0.027 ms], [17.48×],
  [Mean test classifier head], [0.469 ms], [0.027 ms], [17.45×],
  [Mean train backward], [0.722 ms], [0.722 ms], [1.00×],
  [Mean logged train forward + backward], [1.360 ms], [0.915 ms], [1.49×],
), caption: [Jetson CUDA-event timing comparison])

The dense classifier remains the largest component of the original Jetson forward pass, but it is less dominant than in the first timing run. It accounts for 73.61% of the logged training forward time on Jetson, compared with about 90% in the earlier table. After the FC-as-Conv2D rewrite, convolution becomes the dominant Jetson forward component because the classifier head falls to only 0.027 ms.

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  align: (left, right, right, right, right),
  [*Forward component*],
  [*Original FC train mean ms*],
  [*Original FC train share*],
  [*FC-as-Conv2D train mean ms*],
  [*FC-as-Conv2D train share*],
  [Conv], [0.139], [21.80%], [0.137], [70.95%],
  [ReLU], [0.013], [2.05%], [0.013], [6.77%],
  [Max pool], [0.009], [1.42%], [0.009], [4.71%],
  [Classifier head], [0.470], [73.61%], [0.027], [13.93%],
  [Softmax + prediction], [0.007], [1.03%], [0.007], [3.45%],
), caption: [Jetson training forward-pass component breakdown])

The forward-only Jetson speedup is large because the classifier-head kernel is replaced by a more parallel reduction over each `(batch, class)` output. The full logged training compute speedup is smaller because backpropagation is still the original dense-FC backward path. Therefore, on Jetson the FC-as-Conv2D rewrite is most valuable for inference-like forward execution and for diagnosing the forward bottleneck; it does not yet optimize the training backward pass.

== Jetson interpretation

The Jetson measurements strengthen the system-level conclusion of the report. The mathematical network remains shallow and its final accuracy remains around 52%, so the accuracy limit is still dominated by representation capacity and training choices. The hardware-dependent part is the timing distribution: on Jetson, the original classifier head consumes most of the forward pass, while the FC-as-Conv2D equivalent shifts the bottleneck back to the first convolution. This is the desired result of the rewrite, because convolution is the actual image-processing operation that should dominate a small CNN forward pass after removing the inefficient dense-head implementation.

The measured Jetson forward times are CUDA-event times around GPU kernels. They still exclude CPU-side file opening, image decoding, resizing, normalization, and host-to-device transfer. Consequently, the Jetson deployment conclusion is the same as for the other CUDA timing run: the next end-to-end optimization should not be another classifier-head rewrite, but a data path designed for embedded throughput, such as a cached tensor file, pinned host buffers, and overlapped transfer/compute through CUDA streams.


= Limitations

The implementation is clear and useful as a CUDA CNN baseline, but the following image-processing and learning limitations explain the modest final accuracy:

- The network has only one convolutional layer, so its learned representation is mostly low-level.
- There is no data augmentation, so the model is not explicitly trained for common image transformations.
- There is no per-channel standardization, which may slow optimization and make color statistics harder to learn.
- In the original FC implementation, the flatten size is large relative to the network depth, causing most parameters and most measured forward time to reside in the classifier. The FCN-equivalent head removes this measured forward bottleneck, but it does not add representation capacity.
- No confusion matrix or per-class accuracy is logged, so the report cannot identify which visual categories are most confused.
- The CUDA data pipeline reopens and decodes individual image files during batch loading. Because CIFAR-10 is naturally available as compact binary batches, this image-file representation creates avoidable CPU I/O and decoding overhead.
- The training loop uses plain SGD with a fixed learning rate; no momentum, adaptive optimizer, or learning-rate schedule is used.
- The FCN executable demonstrates forward-path equivalence only. Its backward/update path still uses the original dense-FC gradient code, and the current classifier kernel is tied to the fixed $16 times 16$ pooled feature geometry.
- The Jetson logs used in the new embedded-platform section do not include the exact Jetson model, power profile, or clock-lock status, so the Jetson measurements should not be interpreted as a reproducible per-board benchmark without recording those settings.

= Recommendations

A stronger image-processing CNN for CIFAR-10 should keep the efficient CUDA structure but change the representation:

1. Replace the image-folder loader with a CIFAR-10 binary reader or a one-time preprocessing cache that stores contiguous normalized tensors and labels. This is the most important end-to-end performance fix for the current CUDA implementation.
2. Add asynchronous prefetching, pinned host memory, and CUDA streams so that batch loading, host-to-device transfer, and GPU computation can overlap.
3. Add convolutional blocks: for example, `(conv -> ReLU -> conv -> ReLU -> pool)` repeated two or three times.
4. Add per-channel normalization using training-set RGB means and standard deviations.
5. Add random crop with padding, horizontal flip, and light color augmentation.
6. Keep the FC-as-Conv2D forward head or replace the original custom FC path with cuBLAS; then reduce the remaining classifier dependence with additional pooling, global average pooling, or a smaller hidden layer.
7. Use momentum SGD or Adam and a learning-rate schedule.
8. Log a confusion matrix and per-class precision/recall to connect performance errors to image content, such as animals versus vehicles.
9. Consider cuBLAS for the fully connected matrix multiply and fuse simple elementwise kernels where possible.

= Conclusion

The implemented network is a compact CUDA CNN for CIFAR-10. It correctly follows the image-processing pattern of local filtering, non-linear activation, local pooling, and global classification. The experiment demonstrates learning: test accuracy rises from 32.85% after epoch 1 to 51.75% after epoch 10 in the first CUDA run, and the NVIDIA Jetson run reaches 52.41% test accuracy after epoch 10. The comparison with the Python/Keras LeNet shows that the Python model reaches a higher final test accuracy of 58.89%, mainly because it uses a deeper feature hierarchy, Adam, and an in-memory CIFAR-10 data path. The fully convolutional comparison shows that the CUDA classifier can be rewritten as an equivalent valid convolution: the equivalence check reports zero prediction mismatches on a batch of 16, and the final test accuracy remains effectively unchanged. In the first timing run, mean logged training forward time drops from 8.164 ms to 0.989 ms; on the Jetson run, it drops from 0.638 ms to 0.193 ms, with the classifier-head stage falling from 0.470 ms to 0.027 ms. The CUDA implementation's most important end-to-end performance limitation is still the repeated reading and decoding of individual image files. The best next step is therefore twofold: keep the transparent CUDA kernel structure and the faster FC-as-Conv2D head, but replace the input pipeline with a binary or cached tensor loader, then deepen the convolutional feature extractor and reduce dependence on the global $16 times 16$ classifier kernel.

#pagebreak()
#bibliography("bibliography.bib", full: true)

#pagebreak()
#show: appendix
= Reproducibility notes

- Dataset root expected by the executable: `dataset/train/<class-name>` and `dataset/test/<class-name>`.
- Class names in the source code: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
- Main training constants: `BATCH_SIZE = 16`, `EPOCHS = 10`, `FIXED_SEED = 123`, `learningRate = 0.001`, `lambda = 1e-4`.
- Tensor sizes from the run: input `3×32×32`, convolution output `32×32×32`, pool output `32×16×16`, flatten size `8192`, output classes `10`.
- The timing tables in this report are parsed from logged CUDA events and do not include all CPU-side dataset decoding, preprocessing, and host-device transfer costs.
- The Python/Keras comparison uses `lenet.py` and `python_dump.txt`. Its timing is Keras progress-log timing with CIFAR-10 already loaded into memory, so it should not be treated as a kernel-by-kernel comparison with the CUDA event timings.
- The FCN comparison uses `kernels_fcn.cu`, `lenet_fcn.cu`, `main_fcn.cu`, `check_fcn_equivalence.cu`, and `fcn_dump.txt`. The equivalence check reports maximum logit difference $1.13248825 times 10^(-6)$, maximum softmax difference $8.94069672 times 10^(-8)$, and 0/16 prediction mismatches.
- For the CUDA implementation, the current image-file dataset layout is expected to be much slower than using the original CIFAR-10 binary batches or a preprocessed tensor cache.
- The Jetson section was parsed from `jetson_dump.txt` and `jetson_fcn_dump.txt`. Both logs contain 3760 forward timing lines and 3130 training backward timing lines. The final Jetson summary is: train loss 1.3785, train accuracy 52.32%, test loss 1.3795, and test accuracy 52.41%. The logs do not include Jetson model, power mode, or locked-clock configuration.
- *Link to repository:* #link("https://github.com/giulioandrea/Eliva2026")[Github - Eliva2026]
