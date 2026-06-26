# CIFAR-10 CUDA CNN Typst Report

Files:
- `cifar10_cuda_cnn_report.typ`: main Typst report.
- `loss_curve.png`, `accuracy_curve.png`, `timing_breakdown.png`, `forward_time_by_epoch.png`: figures used by the report.
- `epoch_metrics.csv`, `timing_summary.csv`: parsed metrics from `dump.txt`.
- `parsed_stats.json`: machine-readable parsed summary.

Compile locally with:

```bash
typst compile cifar10_cuda_cnn_report.typ cifar10_cuda_cnn_report.pdf
```

The current execution environment did not have the `typst` executable installed, so only the source package is provided.
