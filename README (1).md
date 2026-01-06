# PatternMind 🧠🖼️  
**Supervised image classification + unsupervised pattern discovery (feature-space clustering)**

PatternMind is a machine learning project focused on **multi-class image classification** and **unsupervised pattern discovery**.  
It combines supervised deep learning models (**CNN + ANN baseline**) with clustering and visualization tools (**feature embeddings, silhouette score, t‑SNE**) to understand both *performance* and *data structure*.

---

## What this project does

### 1) Supervised learning (classification)
- Train neural networks to classify images into many categories (hundreds of classes).
- Compare **optimizers** and track training/validation curves (accuracy + loss).
- Evaluate on the **test set** using a **confusion matrix**.

### 2) Unsupervised learning (pattern discovery)
- Reuse the trained classifier as a **feature extractor**.
- Cluster the learned embeddings to discover structure (and potential dataset issues).
- Visualize clusters using **t‑SNE** and measure separation via **silhouette score**.

---

## Dataset

- Images are stored as **NumPy arrays** with shape **(H, W, C)**.
- Labels are categorical and encoded using **LabelEncoder**.
- The class names visible in the confusion matrix correspond to an **object-category dataset similar to Caltech‑256** (hundreds of categories such as *airplanes, americanflag, backpack, …*).

### Typical preprocessing
- Resize to a fixed resolution (e.g., 128×128 or 224×224)
- Normalize pixel values (e.g., `/255.0`)
- Encode labels:
  - `LabelEncoder` → integer labels  
  - Optional one‑hot encoding for softmax classifiers

> If you're using a different dataset, keep the same pipeline: `(H,W,C)` arrays + categorical labels.

---

## Models

### CNN Classifier (main model)
A convolutional network trained with categorical cross‑entropy, used for:
- Final class prediction (softmax)
- Feature extraction (Dense embeddings)

### ANN Baseline (reference model)
A simpler MLP/ANN trained on flattened images (or simple features) as a baseline.

---

## Optimizer Benchmark (what the plots show)

Eight optimizers were compared:

- `Adam`, `SGD`, `RMSprop`, `Adadelta`, `Adagrad`, `Adamax`, `Nadam`, `Ftrl`

**Observed outcome (from the provided curves):**
- **Adamax** achieved the best validation accuracy (≈ **0.22–0.23** after ~60 epochs).
- **Adam / SGD / RMSprop / Nadam** clustered around ≈ **0.18–0.19** validation accuracy.
- **Adagrad** was noticeably lower (≈ **0.11–0.12**).
- **Adadelta** and **Ftrl** underperformed (≈ **0.07–0.08** and ≈ **0.04**).

This suggests the task is **hard + high-class-count**, and optimization choice matters.

---

## Training Dynamics (loss/accuracy plots)

Two training runs are shown:

- Run A: training accuracy rises to ~**0.65**, validation accuracy to ~**0.46**  
  → **generalization gap** appears after ~30 epochs (classic overfitting signal).
- Run B: training accuracy rises to ~**0.34**, validation accuracy to ~**0.23**  
  → slower learning + earlier plateau.

**Practical takeaway:** use **early stopping**, **regularization**, and **data augmentation** (see Improvements).

---

## Evaluation

### Confusion Matrix (Test Set)
A full multi-class confusion matrix is produced for the test set.

Because there are **many classes**, the matrix is mostly sparse; useful things to check:
- whether errors concentrate in “similar” categories
- whether a few classes dominate predictions (dataset imbalance or collapse)
- whether some classes are never predicted (data scarcity / labeling issues)

---

## Feature Extraction Model (Embedding Model)

This model reuses the trained classifier network to extract high‑level feature representations instead of final class predictions.

- **Input:** Same input tensor as the original classifier  
- **Output:** Activations from the `Dense(256)` layer (`layers[-3]`)  
- **Purpose:** Generate a **256‑dimensional feature embedding** for each input sample  

This is commonly used for:
- Transfer learning
- Clustering and similarity analysis
- Downstream ML tasks
- Representation learning and visualization

---

## Unsupervised Clustering Results

### Silhouette plot
- Mean silhouette appears around **0.1** → clusters exist but are **weakly separated**.

### t‑SNE visualization
- The embedding space shows **two visible clusters** plus a noticeable amount of **noise/outliers**.
- This pattern is typical when using density-based clustering (e.g., **DBSCAN/HDBSCAN**) on high-dimensional embeddings.

Interpretation:
- The classifier’s embedding space captures *some* structure,
- but the dataset likely contains overlapping classes, intra-class diversity, or limited feature separability at this stage.

---

## Project Structure (recommended)

You can adapt this to match your repo, but this layout keeps things clean:

```
patternmind/
├── data/                       # optional: local data (usually gitignored)
├── notebooks/                  # experiments + EDA
├── src/
│   ├── train.py                # training entrypoint
│   ├── models.py               # CNN/ANN definitions
│   ├── evaluate.py             # confusion matrix + metrics
│   ├── features.py             # embedding extraction
│   ├── cluster.py              # clustering + silhouette
│   └── visualize.py            # t-SNE + plots
├── outputs/
│   ├── plots/                  # accuracy/loss curves, silhouette, t-SNE
│   └── confusion_matrices/
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train a classifier
Example (optimizer = Adamax):
```bash
python -m src.train --model cnn --optimizer adamax --epochs 60
```

### 3) Evaluate on the test set
```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.keras
```

### 4) Extract embeddings + cluster
```bash
python -m src.features --checkpoint outputs/checkpoints/best_model.keras --layer dense_256
python -m src.cluster --embeddings outputs/embeddings.npy
python -m src.visualize --embeddings outputs/embeddings.npy --labels outputs/cluster_labels.npy
```

> If your project is notebook-based, mirror these steps as notebook sections.

---

## Reproducibility Notes

To make results consistent:
- fix random seeds (NumPy, TensorFlow/PyTorch)
- log configuration (optimizer, LR, batch size, image size)
- save:
  - best checkpoint
  - training history
  - label encoder mapping (class ↔ id)

---

## Recommended Improvements (next power-ups 🚀)

If you want a clean jump in validation accuracy, do these first:

1. **Transfer learning**
   - Use a pretrained backbone (MobileNetV2, EfficientNet, ResNet) and fine-tune.
2. **Data augmentation**
   - Random flips, rotations, crops, color jitter.
3. **Class imbalance handling**
   - Class weights or balanced sampling.
4. **Regularization**
   - Dropout, L2 weight decay, batch norm.
5. **Learning-rate scheduling**
   - ReduceLROnPlateau or cosine decay.
6. **Early stopping**
   - Stop when validation loss stops improving.

---

## Tech Stack

- Python
- NumPy
- scikit-learn (LabelEncoder, confusion matrix, silhouette score)
- TensorFlow/Keras (or PyTorch — depending on your implementation)
- Matplotlib (plots)
- t‑SNE (scikit-learn or openTSNE)

---

## License
Choose one (MIT is a common default). Add a `LICENSE` file if needed.

---

## Acknowledgements
- Dataset inspiration: large-scale object-category benchmarks (Caltech‑256-like)
- Classic tools: scikit-learn + deep learning frameworks for feature learning and evaluation
