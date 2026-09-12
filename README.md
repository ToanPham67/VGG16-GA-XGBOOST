# A Hybrid Model for Image Classification Using Optimal VGG16 Features and XGBoost Classifier
---
## 1. VGG16 Image Feature Extraction Pipeline

This repository contains an end-to-end Python pipeline for extracting deep feature representations from medical image datasets (e.g., LC25000) using a pretrained **VGG16 Convolutional Base** combined with **Global Average Pooling (GAP)**.

---

##  Pipeline Overview

1. **Dataset Ingestion:** Automatically scans directory structures, maps target classes, and indexes image file paths.
2. **Preprocessing:** Resizes images to $224 \times 224$ pixels, converts them to PyTorch Tensors, and applies ImageNet normalization.
3. **Deep Feature Extraction:**
   * Passes inputs through the VGG16 convolutional backbone.
   * Applies `AdaptiveAvgPool2d` to collapse spatial dimensions into a compact $512$-dimensional feature vector per image.
4. **Fault Tolerance:** Catches unreadable or corrupted images during loading without interrupting the batch execution flow.
5. **Persistence:** Exports extracted features, encoded labels, and valid image paths to disk in NumPy format (`.npy`).

---

##  Requirements & Installation

- **Python:** 3.8+
- **Hardware:** CUDA-compatible GPU recommended for optimal extraction speed.

Install dependencies via `pip`:

```python
pip install torch torchvision numpy pillow opencv-python tqdm
```
---




