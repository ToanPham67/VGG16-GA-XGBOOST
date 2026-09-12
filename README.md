# A Hybrid Model for Image Classification Using Optimal VGG16 Features and XGBoost Classifier

- **Stage 1. Feature Extraction**: Use VGG16 to extract deep image features.
- **Stage 2. Feature Selection**: Apply a Genetic Algorithm (GA) to select the most informative features and reduce redundancy.
- **Stage 3. Classification**: Use XGBoost to classify the selected features.
- **Evaluation**: Evaluate the proposed model on two image datasets in terms of accuracy and computational efficiency.

| The proposed model
| :---: 
| ![The flowchart of the proposed model](GV-trainMRI.png)

---
## 1. VGG16 Image Feature Extraction Pipeline

This repository contains an end-to-end Python pipeline for extracting deep feature representations from medical image datasets (e.g., LC25000) using a pretrained **VGG16 Convolutional Base** combined with **Global Average Pooling (GAP)**.

##  Pipeline Overview

1. **Dataset Ingestion:** Automatically scans directory structures, maps target classes, and indexes image file paths.
2. **Preprocessing:** Resizes images to $224 \times 224$ pixels, converts them to PyTorch Tensors, and applies ImageNet normalization.
3. **Deep Feature Extraction:**
   * Passes inputs through the VGG16 convolutional backbone.
   * Applies `AdaptiveAvgPool2d` to collapse spatial dimensions into a compact $512$-dimensional feature vector per image.
4. **Fault Tolerance:** Catches unreadable or corrupted images during loading without interrupting the batch execution flow.
5. **Persistence:** Exports extracted features, encoded labels, and valid image paths to disk in NumPy format (`.npy`).

##  Requirements & Installation

- [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
- **Hardware:** CUDA-compatible GPU recommended for optimal extraction speed.

Install dependencies via `pip`:

```python
pip install torch torchvision numpy pillow opencv-python tqdm
```
---
##  Repository Structure
```text
.
├── features_VGG16.npz            # VGG16 feature extraction
├── VGG16_GV_XGBoost.ipynb        # Jupyter Notebook
├── GA_selected_features_MRI.png  # Number of Selected Features Plot
├── Results                       # Output plots
├── requirements.txt              # Required dependencies
└── README.md                     # Documentation
```
## Dataset Structure

This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format. The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package. There are five classes in the dataset, each with 5,000 images, being: (kaggle: https://www.kaggle.com/datasets/javaidahmadwani/lc25000)

Links https://arxiv.org/abs/1912.12142v1 https://github.com/tampapath/lung_colon_image_set Dataset BibTeX @article{, title= {LC25000 Lung and colon histopathological image dataset}, keywords= {cancer,histopathology}, author= {Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, Stephen M. Mastorides}, url= {https://github.com/tampapath/lung_colon_image_set} }

```text
LC25000/
└── lung_colon_image_set/
    ├── Test Set/
    │   ├── colon_aca/
    │   ├── colon_n/
    │   ├── lung_aca/
    │   ├── lung_n/
    │   └── lung_scc/
    │
    └── Train and Validation Set/
        ├── colon_aca/
        ├── colon_n/
        ├── lung_aca/
        ├── lung_n/
        └── lung_scc/
```

##  Usage Guide

###  Case 1: End-to-End Extraction & GA Selection
Extract 512-dimensional deep feature representations from raw images using the pre-trained VGG16 model and optimize feature subsets via a parallel Genetic Algorithm (GA).

1. **Run Cell 1 (Feature Extraction)**:
   - Traverses the `Train and Validation Set` and `Test Set` directories.
   - Preprocesses images to **224 × 224** and extracts **512-dimensional feature embeddings** using the pre-trained VGG16 model.
   - Automatically saves the extracted features to `features_VGG16.npz` for caching and reuse.
```python
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import warnings
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
# Start timing
st = time.time()
warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, label_map):
        self.image_paths = image_paths
        self.labels = labels
        self.label_map = label_map
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # VGG16 uses 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.preprocess(image)
            label_idx = self.label_map[label]
            return image_tensor, label_idx, img_path
        except (UnidentifiedImageError, Exception) as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224), -1, img_path  # Dummy for invalid image
def extract_features(image_paths, labels, device=None,
                     batch_size=16, num_workers=4, save_features=False,
                     output_dir='features', label_map=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(image_paths) != len(labels):
        raise ValueError("Number of image paths and labels must match.")
    if len(image_paths) == 0:
        print("No images provided.")
        return np.array([]), np.array([]), []
    # Load pretrained VGG16
    model = models.vgg16(pretrained=True)
    model = model.features  # Use only the convolutional base
    model.eval().to(device)
    dataset = ImageDataset(image_paths, labels, label_map=label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False,
                            pin_memory=(device != 'cpu'))
    features, extracted_labels, valid_paths = [], [], []
    for batch_tensors, batch_labels, batch_paths in tqdm(dataloader, desc="Extracting features"):
        batch_tensors = batch_tensors.to(device)
        valid_mask = batch_labels != -1
        if valid_mask.sum() == 0:
            continue
        with torch.no_grad():
            # Pass through VGG16 convolutional layers
            conv_feats = model(batch_tensors[valid_mask])
            # Apply global average pooling to get [B, 512]
            batch_feat = torch.nn.functional.adaptive_avg_pool2d(conv_feats, (1, 1)).view(conv_feats.size(0), -1)
            batch_feat = batch_feat.cpu().numpy()
        features.append(batch_feat)
        extracted_labels.extend(batch_labels[valid_mask].cpu().numpy())
        valid_paths.extend([batch_paths[i] for i in range(len(batch_paths)) if valid_mask[i]])
    if not features:
        print("No valid images processed.")
        return np.empty((0, 512)), np.array([]), []
    features = np.concatenate(features, axis=0)
    extracted_labels = np.array(extracted_labels)
    if save_features:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'features.npy'), features)
        np.save(os.path.join(output_dir, 'labels.npy'), extracted_labels)
        with open(os.path.join(output_dir, 'valid_paths.txt'), 'w') as f:
            f.write('\n'.join(valid_paths))
        print(f"Features saved to {output_dir}")
    print(f"Extracted {features.shape[0]} features.")
    return features, extracted_labels, valid_paths
def get_image_paths_and_labels(folder_path):
    image_paths, labels, label_map = [], [], {}
    for idx, label_name in enumerate(sorted(os.listdir(folder_path))):
        label_folder = os.path.join(folder_path, label_name)
        if os.path.isdir(label_folder):
            label_map[label_name] = idx
            for filename in os.listdir(label_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(label_folder, filename))
                    labels.append(label_name)
    return image_paths, labels, label_map
# Folder paths
train_folder = "Train and Validation Set"
test_folder = "Test Set"
# Load image paths and labels
x_train_paths, y_train, train_label_map = get_image_paths_and_labels(train_folder)
x_test_paths, y_test, test_label_map = get_image_paths_and_labels(test_folder)
# Confirm device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Extract test features
print("Extracting features for test set...")
xtest_features, y_test_labels, _ = extract_features(
    x_test_paths, y_test, device=device, batch_size=16,
    num_workers=4, save_features=True,
    output_dir='test_features', label_map=test_label_map)
# Extract train features
print("Extracting features for training set...")
xtrain_features, y_train_labels, _ = extract_features(
    x_train_paths, y_train, device=device, batch_size=16,
    num_workers=4, save_features=True,
    output_dir='train_features', label_map=train_label_map)
# Output shapes
print(f"Train features shape: {xtrain_features.shape}")
print(f"Train labels shape: {y_train_labels.shape}")
print(f"Test features shape: {xtest_features.shape}")
print(f"Test labels shape: {y_test_labels.shape}")
# End timing
end = time.time()
print(f"Total execution time: {end - st:.2f} seconds")
```
2. **Run Cell 2: GA (Parallel) without K-Fold Cross-Validation**:
   - Selects the optimal subset of extracted features using parallel fitness evaluations with K-Fold Cross-Validation.
   - Fits a Logistic Regression classifier on the selected training features.
```python
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
start_time = time.time()
# ---------------------------------------------------------
# 1. Data Preparation & Device Setup
# ---------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    xtrain_features,
    y_train_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_train_labels,)
# Convert split data to PyTorch Tensors
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
# Multi-GPU setup (default to CUDA:0 or CPU if unavailable)
device_count = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(device_count)] if device_count > 0 else [torch.device("cpu")]
print(f"Running on {len(devices)} device(s): {devices}")
# ---------------------------------------------------------
# 2. Fast GPU Batch Evaluation (Ridge Closed-Form Solution)
# ---------------------------------------------------------
def evaluate_batch_gpu(population_subset, device, X_tr_t, y_tr_t, X_val_t, y_val_t):
    """Evaluates a subset of GA individuals on a specified GPU/CPU device."""
    X_tr_d, y_tr_d = X_tr_t.to(device), y_tr_t.to(device).unsqueeze(1)
    X_val_d, y_val_d = X_val_t.to(device), y_val_t.to(device)
    scores = []
    for ind in population_subset:
        selected = torch.tensor(ind, dtype=torch.bool, device=device)
        if not selected.any():
            scores.append(0.0)
            continue
        # Extract selected features and add bias column
        X_tr_sub = torch.cat([torch.ones((X_tr_d.shape[0], 1), device=device), X_tr_d[:, selected]], dim=1)
        X_val_sub = torch.cat([torch.ones((X_val_d.shape[0], 1), device=device), X_val_d[:, selected]], dim=1)
        # Closed-form Ridge Classifier solution: W = (X^T * X + lambda * I)^(-1) * X^T * Y
        reg = 1e-4 * torch.eye(X_tr_sub.shape[1], device=device)
        weights = torch.linalg.pinv(X_tr_sub.T @ X_tr_sub + reg) @ X_tr_sub.T @ (y_tr_d * 2 - 1)
        # Predict and evaluate accuracy on validation set
        preds = (X_val_sub @ weights) > 0
        acc = (preds.squeeze() == y_val_d).float().mean().item()
        scores.append(acc)
    return scores
# ---------------------------------------------------------
# 3. Genetic Algorithm Core Function
# ---------------------------------------------------------
def genetic_algorithm(pop_size=50, generations=200, mutation_rate=0.01):
    n_features = X_tr.shape[1]
    population = [np.random.randint(0, 2, n_features).astype(bool) for _ in range(pop_size)]
    best_history, avg_history, feature_count_history = [], [], []
    best_ind_overall, best_fit_overall = None, -1.0
    for gen in range(generations):
        # Parallel evaluation across available devices
        split_size = pop_size // len(devices)
        scores = []
        for i, dev in enumerate(devices):
            sub_pop = population[i * split_size :] if i == len(devices) - 1 else population[i * split_size : (i + 1) * split_size]
            scores.extend(evaluate_batch_gpu(sub_pop, dev, X_tr_t, y_tr_t, X_val_t, y_val_t))
        scores = np.array(scores)
        best_idx = np.argmax(scores)
        best_score, avg_score = scores[best_idx], np.mean(scores)
        best_ind = population[best_idx].copy()
        # Track history metrics
        best_history.append(best_score)
        avg_history.append(avg_score)
        feature_count_history.append(np.sum(best_ind))
        if best_score > best_fit_overall:
            best_fit_overall = best_score
            best_ind_overall = best_ind.copy()
        print(f"Gen {gen+1:03d} | Best Fitness: {best_score:.4f} | Avg Fitness: {avg_score:.4f} | Features: {np.sum(best_ind)}")
        # Selection: Retain top 50% parents
        sorted_indices = np.argsort(scores)[::-1]
        population = [population[idx] for idx in sorted_indices[: pop_size // 2]]
        # Crossover & Vectorized Mutation to refill population
        while len(population) < pop_size:
            p1, p2 = random.sample(population, 2)
            cp = random.randint(1, n_features - 1)
            child = np.concatenate([p1[:cp], p2[cp:]])
            # Fast numpy mutation
            mutation_mask = np.random.rand(n_features) < mutation_rate
            child[mutation_mask] = ~child[mutation_mask]
            population.append(child)
    return best_ind_overall, best_fit_overall, best_history, avg_history, feature_count_history
# ---------------------------------------------------------
# 4. Execution & Model Evaluation
# ---------------------------------------------------------
print("\n--- STARTING PARALLEL GENETIC ALGORITHM ---")
best_ind, best_fit, best_hist, avg_hist, feat_hist = genetic_algorithm(pop_size=50, generations=200)
# Final validation on Test Set using Scikit-Learn Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(xtrain_features[:, best_ind], y_train_labels)
test_acc = accuracy_score(y_test_labels, clf.predict(xtest_features[:, best_ind]))
print("\n--- FINAL RESULTS ---")
print(f"Best Validation Accuracy (GA): {best_fit:.4f}")
print(f"Final Test Accuracy:          {test_acc:.4f}")
print(f"Selected Features Count:      {np.sum(best_ind)} / {X_tr.shape[1]}")
print(f"Total Execution Time:         {time.time() - start_time:.2f} seconds")
# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------
gen_range = range(1, len(best_hist) + 1)
# Plot 1: Convergence Curve
plt.figure(figsize=(8, 5))
plt.plot(gen_range, best_hist, linewidth=2, label="Best Objective Value")
plt.plot(gen_range, avg_hist, linewidth=2, linestyle="--", label="Average Objective Value")
plt.title("GA Convergence Curve")
plt.xlabel("Generation")
plt.ylabel("Accuracy Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("GA_convergence_curve_MRI.png", dpi=300, bbox_inches="tight")
plt.show()
# Plot 2: Selected Features Trend
plt.figure(figsize=(8, 5))
plt.plot(gen_range, feat_hist, color="green", linewidth=2)
plt.title("Selected Features per Generation")
plt.xlabel("Generation")
plt.ylabel("Number of Features")
plt.grid(True)
plt.tight_layout()
plt.savefig("GA_selected_features_MRI.png", dpi=300, bbox_inches="tight")
plt.show()
```
3. **Run Cell 3: XGBoost with K-Fold Cross-Validation**:
  - Train and Evaluate XGBoost Model using GA-Selected Features.
  - The code below applies the optimal feature set (best_individual) identified by the Genetic Algorithm to train an XGBoost classifier, then outputs the overall test accuracy, learning curves (Log Loss), and Confusion Matrix.
---
**01 repeated 5-fold cross-validation**
```python
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
st = time.time()
# --------- Step 1: Feature Selection via GA ---------
X_train_selected = xtrain_features[:, best_individual]
X_test_selected = xtest_features[:, best_individual]
# --------- Step 2: Configure & Train XGBoost ---------
clf = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
    verbosity=0,
    random_state=42,)
clf.fit(
    X_train_selected,
    y_train_labels,
    eval_set=[(X_train_selected, y_train_labels), (X_test_selected, y_test_labels)],
    verbose=False,)
evals_result = clf.evals_result()
# --------- Step 3: Test Set Prediction ---------
y_pred = clf.predict(X_test_selected)
test_accuracy = accuracy_score(y_test_labels, y_pred)
print(f"\nTest accuracy using selected features (XGBoost): {test_accuracy:.4f}")
# --------- Step 4: Plot Log Loss ---------
epochs = len(evals_result["validation_0"]["mlogloss"])
plt.figure(figsize=(7, 5))
plt.plot(range(epochs), evals_result["validation_0"]["mlogloss"], label="Train")
plt.plot(range(epochs), evals_result["validation_1"]["mlogloss"], label="Test")
plt.xlabel("Epochs")
plt.ylabel("Log Loss (mlogloss)")
plt.title("XGBoost Log Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("XGBoost_mlogloss.png", dpi=300, bbox_inches="tight")
plt.show()
# --------- Step 5: Plot Confusion Matrix ---------
cm = confusion_matrix(y_test_labels, y_pred)
classes = np.unique(y_test_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (XGBoost with GA features)")
plt.tight_layout()
plt.savefig("CM_MRIEX2.png", dpi=300, bbox_inches="tight")
plt.show()
# --------- Step 6: Classification Report ---------
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred))
print(f" Total Time: {time.time() - st:.2f} seconds")
```
---
**30 repeated 5-fold cross-validation**
```python
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
st = time.time()
# --------- Step 1: Feature Selection via GA ---------
X_selected = xtrain_features[:, best_individual]
y_labels = y_train_labels
# --------- Step 2: Configure 30-Repeated 5-Fold Cross-Validation ---------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=42)
accuracies = []
oof_y_true = []
oof_y_pred = []
print("Running 30x5-Fold Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(rskf.split(X_selected, y_labels)):
    X_tr, X_val = X_selected[train_idx], X_selected[val_idx]
    y_tr, y_val = y_labels[train_idx], y_labels[val_idx]
    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        verbosity=0,
        random_state=42,)
    clf.fit(X_tr, y_tr, verbose=False)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    
    oof_y_true.extend(y_val)
    oof_y_pred.extend(y_pred)
# --------- Step 3: Print CV Results ---------
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\nMean CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
# --------- Step 4: Aggregate Confusion Matrix ---------
cm = confusion_matrix(oof_y_true, oof_y_pred)
classes = np.unique(y_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Aggregated Confusion Matrix (30x5-Fold CV)")
plt.tight_layout()
plt.savefig("CM_MRIEX2_30x5Fold.png", dpi=300, bbox_inches="tight")
plt.show()
# --------- Step 5: Classification Report ---------
print("\nOverall Classification Report (Cross-Validation):")
print(classification_report(oof_y_true, oof_y_pred))
print(f"Total Time: {time.time() - st:.2f} seconds")
```
---
###  Case 2: Fast Experimentation via Feature Cache
Skip the time-consuming VGG16 extraction phase by re-loading pre-calculated feature tensors directly into memory.

1. **Run Cell 1 (Cache Loading)**:
* **[CACHE DETECTED]** Found existing `features_LC25000.npz` cache file.
* **[SUCCESS]** Loaded `xtrain_features`, `xtest_features`, `y_train_labels`, and `y_test_labels` arrays in seconds.
```python
import numpy as np
# Load the saved NPZ file
data = np.load("features_LC25000.npz")
xtrain_features = data["xtrain_features"]
y_train_labels = data["y_train_labels"]
xtest_features = data["xtest_features"]
y_test_labels = data["y_test_labels"]
print("Loaded features_LC25000.npz in seconds.")
```
2. **Run Cell 2: GA (Parallel) without K-Fold Cross-Validation**:
   - Run the same code as in Case 1.
3. **Run Cell 3: XGBoost with K-Fold Cross-Validation**:
   - Run the same code as in Case 1.
---
### Evaluation Strategies: K-Fold CV vs. Train/Val Split

You can configure the evaluation strategy inside **Cell** by setting the `USE_KFOLD` parameter:

* **Single Split Mode (`USE_KFOLD = False`)**:
  - Splits training data into an internal 80/20 Train/Validation set once.
  - Recommended for fast GA optimization across high generation counts ($1000+$ generations).

* **Stratified K-Fold CV (`USE_KFOLD = True`)**:
  - Evaluates each individual's fitness using Stratified $K$-Fold Cross-Validation (default: $K=5$).
  - Recommended for robust validation preventing sample split bias.

```python
# Configure in Cell :
USE_KFOLD = 5  # Set to True to enable 5-Fold Stratified CV
```
---
**Continue with the other machine learning models.**
- **Random Forest**
```python
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
st = time.time()
# 1. Feature Selection
X_tr, X_te = xtrain_features[:, best_individual], xtest_features[:, best_individual]
# 2. Train Model
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_train_labels)
# 3. Predict & Evaluate
y_pred = clf.predict(X_te)
print(f"Test Accuracy: {accuracy_score(y_test_labels, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test_labels, y_pred))
# 4. Plot Confusion Matrix
classes = np.unique(y_test_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (RF + GA Features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("CM_RF.png", dpi=300)
plt.show()
print(f"Total Time: {time.time() - st:.2f}s")
```
- **SVM**
```python
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
st = time.time()
# 1. Feature Selection
X_tr, X_te = xtrain_features[:, best_individual], xtest_features[:, best_individual]
# 2. Train SVM Pipeline
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42))])
clf.fit(X_tr, y_train_labels)
# 3. Predict & Evaluate
y_pred = clf.predict(X_te)
print(f"Test Accuracy: {accuracy_score(y_test_labels, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test_labels, y_pred))
# 4. Plot Confusion Matrix
classes = np.unique(y_test_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (SVM + GA Features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("CM_SVM.png", dpi=300)
plt.show()
print(f"Total Time: {time.time() - st:.2f}s")
```

- **Decision Tree**
```python
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
st = time.time()
X_tr, X_te = xtrain_features[:, best_individual], xtest_features[:, best_individual]
clf = DecisionTreeClassifier(criterion="gini", random_state=42)
clf.fit(X_tr, y_train_labels)
y_pred = clf.predict(X_te)
print(f"Test Accuracy: {accuracy_score(y_test_labels, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test_labels, y_pred))
classes = np.unique(y_test_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (DT + GA Features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("CM_DT.png", dpi=300)
plt.show()
print(f"Total Time: {time.time() - st:.2f}s")
```

- **Naive Bayes**
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
st = time.time()
X_train_selected = features_train[:, best_individual]
X_test_selected  = features_test[:, best_individual]
# --------- (Gaussian) ---------
clf = GaussianNB()
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy using selected features (GaussianNB): {test_accuracy:.4f}")
# --------- Confusion Matrix ---------
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y_test)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (GaussianNB with GA features)")
plt.savefig("CM_GNB.png")
plt.show()
# --------- Classification Report ---------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
end = time.time()
print("Total Time:", end - st)
```
- **Adaboost**
```python
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
st = time.time()
X_tr, X_te = xtrain_features[:, best_individual], xtest_features[:, best_individual]
clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=300, learning_rate=0.5, random_state=42)
clf.fit(X_tr, y_train_labels)
y_pred = clf.predict(X_te)
print(f"Test Accuracy: {accuracy_score(y_test_labels, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test_labels, y_pred))
train_err = [1 - accuracy_score(y_train_labels, yh) for yh in clf.staged_predict(X_tr)]
test_err = [1 - accuracy_score(y_test_labels, yh) for yh in clf.staged_predict(X_te)]
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(train_err) + 1), train_err, label="Train Error")
plt.plot(range(1, len(test_err) + 1), test_err, label="Test Error")
plt.xlabel("Number of Estimators")
plt.ylabel("Error Rate")
plt.title("AdaBoost Error over Estimators")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AdaBoost_error.png", dpi=300)
plt.show()
classes = np.unique(y_test_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (AdaBoost + GA Features)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("CM_AdaBoost.png", dpi=300)
plt.show()
print(f"Total Time: {time.time() - st:.2f}s")
```

- **ANN**
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
st = time.time()
X_train_selected = features_train[:, best_individual]
X_test_selected  = features_test[:, best_individual]
# early_stopping=True sẽ tự tách 1 phần train làm validation và dừng sớm khi không cải thiện
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128),   # bạn có thể đổi (128,), (256,128,64)...
        activation="relu",
        solver="adam",
        alpha=1e-4,                      # L2 regularization
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=200,                    # số epoch tối đa
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    ))
])
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy using selected features (ANN-MLP): {test_accuracy:.4f}")
# -------- Plot loss curve (epoch) ---------
mlp = clf.named_steps["mlp"]
loss_curve = getattr(mlp, "loss_curve_", None)
if loss_curve is not None and len(loss_curve) > 0:
    x_axis = range(1, len(loss_curve) + 1)
    plt.figure(figsize=(7,5))
    plt.plot(x_axis, loss_curve, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN (MLP) Training Loss over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig("ANN_loss.png")
    plt.show()
else:
    print("No loss_curve_ found (sklearn version/config).")
# --------- Confusion Matrix ---------
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y_test)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (ANN-MLP with GA features)")
plt.savefig("CM_ANN.png")
plt.show()
# --------- Classification Report ---------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
end = time.time()
print("Total Time:", end - st)
```




