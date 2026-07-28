# Evidence files

- `comparison_metrics_60images.csv`: per-image evaluation records for the comparative test (60 images, 321 ground-truth vessels). Recomputing TP/FP/FN, precision, recall, and F1 for all five model configurations from this file reproduces Table 6 of the thesis exactly. Mean IoU values are detection-weighted averages over matched detections.
- `training_results.csv`: per-epoch training log of the fine-tuning run (50 epochs, NVIDIA L4). Final row: mAP@0.5 = 0.8406, mAP@0.5:0.95 = 0.6268, precision = 0.788, recall = 0.784, total time 745 s (12.4 min).

Both files were exported from the original Colab runs (December 2025). Originals remain on the author's Google Drive.
