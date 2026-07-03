# Eliva2026

The default setting is that the dataset is in the src/ directory.
If the dataset isn't in the src/ directory, it must be passed as an argument to the program.

example: 
- make all
- ./main /path/to/dataset
or
- make all
- ./main

# Fully-convolutional additions for the LeNet/CUDA codebase

These files add a second forward path without modifying the original source files.

## What changes

The original classification head is:

```text
MaxPool output: [B, 32, 16, 16]
Flatten:        [B, 8192]
FC weights:     [8192, 10]
Logits:         [B, 10]
```

The additive fully-convolutional head is:

```text
MaxPool output: [B, 32, 16, 16]
Conv2D weight:  [10, 32, 16, 16]
Logits map:     [B, 10, 1, 1] -> stored as [B, 10]
```

The mapping is exact:

```c
flatten_index = (channel * POOL_OUTPUT_H + y) * POOL_OUTPUT_W + x;
conv_weight[class][channel][y][x] = fc_weight[flatten_index][class];
```

No retraining is required for inference equivalence. During training, the supplied
`main_fcn.cu` keeps the original backward pass untouched and simply syncs the
reshaped Conv2D view after FC weights change.

## Added files

- `kernels_fcn.h` / `kernels_fcn.cu`: FC-to-Conv2D reshape and FC-as-Conv2D CUDA kernels.
- `lenet_fcn.h` / `lenet_fcn.cu`: wrapper around the original `LeNet` struct.
- `main_fcn.cu`: second executable using the fully-convolutional forward path.
- `check_fcn_equivalence.cu`: small numerical check against the original `LeNet_forward`.
- `Makefile.fcn`: example build file that compiles the original files plus these additions.

## How to use

Copy these files next to the original source files, then build:

```bash
make -f Makefile.fcn
```

Run the equivalence check:

```bash
./check_fcn_equivalence
```

Run the FCN training/evaluation executable:

```bash
./lenet_fcn_app dataset
```

## Important caveat

For the current 32x32 CIFAR-shaped input, this FC-as-Conv2D conversion has the
same number of learned parameters and multiply-adds as the FC layer:

```text
FC:     8192 * 10 = 81,920 weights / MACs per image
Conv2D: 10 * 32 * 16 * 16 = 81,920 weights / MACs per image
```

So this is the correct no-retraining experiment requested, but it
should not be presented as a parameter/MAC reduction by itself. To reduce the
classification-head footprint, use a smaller head such as global average pooling plus
1x1 convolution, but that changes the architecture and requires retraining.


## Collaborators

- Luca Bertetto
- Giulio Andrea Quaglia
