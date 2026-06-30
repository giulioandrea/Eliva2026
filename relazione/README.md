# CUDA CNN report on CIFAR-10

This package contains the revised Typst report with an added comparison between:

- the pure CUDA/C implementation, which loads CIFAR-10 from individual image files, and
- the Python/Keras LeNet baseline in `lenet.py`, which loads CIFAR-10 into memory through `datasets.cifar10.load_data()`.

The report emphasizes that the CUDA event timings measure GPU kernels only. The end-to-end CUDA bottleneck is the CPU-side file reading/decode/resize pipeline because each batch is built by reopening and decoding individual PNG/JPEG files.

Main files:

- `cifar10_cuda_cnn_report.typ`: report source to compile.
- `cifar10_cuda_cnn_report_with_python_comparison.typ`: same report under a descriptive name.
- `loss_curve.png`, `accuracy_curve.png`, `timing_breakdown.png`, `forward_time_by_epoch.png`: figures used by the report.
- `epoch_metrics.csv`, `timing_summary.csv`, `python_lenet_metrics.csv`: parsed metrics.
- `lenet.py`, `python_dump.txt`: Python/Keras baseline source and output log.
- `bibliography.bib`: citation file for the CIFAR-10 dataset page.

Compile locally with:

```bash
typst compile cifar10_cuda_cnn_report.typ cifar10_cuda_cnn_report.pdf
```

The report imports Typst preview packages (`cetz`, `cetz-plot`, and `codly`). A local Typst installation with package access is required.
