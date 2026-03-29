# Task Vectors — Installation and Usage Guide

This repository contains code for exploring **task vectors** applied to Vision Transformer (ViT) models on datasets such as GTSRB, SVHN, MNIST, and others.
The main analysis script is `test_negate.py`, which implements a multi-phase experimental pipeline.

This guide provides detailed instructions on how to:

* set up the environment,
* manage checkpoints and datasets,
* run the experiments.

---

## Overview

**Requirements**

* Conda (or Miniconda)
* Python (as specified in `environment.yml`)

**Main components**

* Environment: configured via `environment.yml`
* Checkpoints: not included (must be downloaded manually)
* Datasets: support automatic download or manual setup

---

## Project Structure

```
.
├── test_negate.py            # Main script (multi-phase pipeline)
├── src/                     # Shared modules (datasets, utils, evaluation)
├── checkpoints/             # Pretrained and finetuned models (not included)
├── Desktop/neuraln/task_vectors/data$/  # Default dataset path
├── environment.yml          # Conda environment configuration
```

---

## 1. Environment Setup

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate neuraln
```

## 2. Checkpoints

Required `.pt` files are **not included** in the repository.

### Expected structure

```
checkpoints/
└── ViT-B-16/
    ├── zeroshot.pt
    ├── GTSRB/
    │   └── finetuned.pt
    └── SVHN/
        └── finetuned.pt
```

### How to obtain them

You can retrieve checkpoints via:

* external repositories
* cloud storage (Google Drive, S3, Hugging Face)
* dedicated download scripts (recommended)

👉 Official download link (from the reference paper):
https://github.com/2076864/NeuralNetwork/blob/main/README.md#:~:text=Checkpoints%20for%20CLIP,Download%20here

## 3. Datasets

### Default path

```python
args.data_location = 'Desktop/neuraln/task_vectors/data$'
```

### Automatic download

Many datasets (e.g., MNIST, SVHN, GTSRB) are automatically downloaded if not found locally.

### Manual download

Alternatively, datasets can be downloaded manually using the same sources provided in the reference paper:

👉 https://github.com/2076864/NeuralNetwork/blob/main/README.md#:~:text=Checkpoints%20for%20CLIP,Download%20here

and placed in the configured directory.

### Custom dataset path

```bash
python test_negate.py --data-location /path/to/datasets
```

or modify `test_negate.py` directly.

---

## 4. Running Experiments

The script `test_negate.py` is organized into **8 experimental phases**, controlled via boolean flags:

| Phase | Flag                     |
| ----- | ------------------------ |
| 0     | `RUN_BASELINE`           |
| 1     | `RUN_ONLY_GTSRB`         |
| 2     | `RUN_ONLY_SVHN`          |
| 3     | `RUN_MULTI`              |
| 4     | `RUN_WEIGHTED`           |
| 5     | `RUN_NEGATION_GTSRB`     |
| 6     | `RUN_NEGATION_SVHN`      |
| 7     | `RUN_NEGATION_BASE`      |
| 8     | `RUN_NEGATION_MULTITASK` |

## 5. Recommended .gitignore

```
# checkpoints
checkpoints/
*.pt
*.pth

# datasets
Desktop/neuraln/task_vectors/data$/
data$/
data/

# python
__pycache__/
*.pyc
.env

# logs
*.log
```

---

