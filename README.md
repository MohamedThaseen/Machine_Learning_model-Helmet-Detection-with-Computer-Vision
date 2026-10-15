# HelmNet: Safety Helmet Detection with Computer Vision

An image classification project for **SafeGuard Corp** that automatically detects whether a worker is wearing a safety helmet. The goal is a model that can be deployed in an automated monitoring system to support workplace safety compliance on construction sites and in industrial plants.

## Table of Contents
- [Business Context](#business-context)
- [Objective](#objective)
- [Dataset](#dataset)
- [Approach](#approach)
- [Models](#models)
- [Results](#results)
- [Caveats and Limitations](#caveats-and-limitations)
- [Business Recommendations](#business-recommendations)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

## Business Context
Safety helmets protect workers from head injuries caused by falling objects and machinery. Manual monitoring of helmet compliance is error-prone and does not scale in large operations. An automated image analysis system improves enforcement, efficiency, scalability and accuracy while reducing human error.

## Objective
Build a binary image classifier that labels each image as:

| Label | Class | Meaning |
|-------|-------|---------|
| `1` | With Helmet | Worker is wearing a safety helmet |
| `0` | Without Helmet | Worker is not wearing a safety helmet |

## Dataset
- **631 RGB images**, each **200 × 200 × 3** (`images_proj.npy`), with labels in `Labels_proj.csv` (one `Label` column).
- **311** with helmet, **320** without helmet, so the classes are nearly balanced.
- Images cover construction sites, factories and industrial settings, with variation in lighting, angles and worker postures (standing, using tools, moving).

> The dataset files are not included in this repository. Place them in a local `data/` folder (see [Getting Started](#getting-started)).

## Approach
1. **Data overview and EDA**: shape checks, sample image grids, class balance (count plot).
2. **Pre-processing**
   - RGB to grayscale conversion (used for a comparison run of the base CNN).
   - Stratified split: **80% train (504) / 10% validation (63) / 10% test (64)**.
   - Pixel normalisation (`/255`).
3. **Compile-setting exploration**: Adam, SGD and Adagrad with `categorical_crossentropy` stayed at about 50% accuracy. Switching to a single sigmoid output with `binary_crossentropy` fixed the learning problem, so the later models use this combination.
4. **Model building and comparison**: base CNN, then three VGG-16 transfer-learning variants.
5. **Evaluation**: accuracy, recall, precision, F1 and confusion matrices on train, validation and test sets.

## Models
| # | Model | Notes |
|---|-------|-------|
| Base | Simple CNN (3 Conv + MaxPool blocks, Dense 100, sigmoid) | Trained on RGB and on grayscale |
| 1 | VGG-16 (frozen, ImageNet) + sigmoid head | Transfer learning baseline |
| 2 | VGG-16 + FFNN (Dense 256, Dropout 0.4, Dense 32) | |
| 3 | VGG-16 + FFNN + **data augmentation** | Rotation, shifts, shear and zoom; **selected as final model** |

## Results
Metrics as reported in the notebook (Accuracy / Recall / Precision / F1):

| Model | Train | Validation |
|-------|-------|------------|
| Base CNN (RGB, categorical-crossentropy setup) | 50.9% accuracy | 50.8% accuracy |
| VGG-16 + sigmoid head | 100% across all metrics | 100% across all metrics |
| VGG-16 + FFNN | 50.8% accuracy | 50.8% accuracy |
| **VGG-16 + FFNN + Augmentation** | **100%** | **100%** |

**Final model on the held-out test set (64 images): 100% accuracy, recall, precision and F1.**

The base CNN and the VGG-16 + FFNN model used `loss='categorical_crossentropy'` with a single sigmoid unit. That loss is a mismatch for one output node, and both models collapsed to roughly chance level (their training loss shows as 0.0). The scores above reflect that setup, not the ceiling of those architectures.

## Caveats and Limitations
- **Perfect scores on a small test set need scrutiny.** 100% on 64 test images is encouraging but statistically fragile. A single mislabelled or duplicate image, or near-duplicate frames split across train and test, can inflate results.
- **Possible data leakage.** The dataset is small and likely contains similar images. Verify that no near-duplicates span the train, validation and test splits.
- **Validation setup is uneven.** Some runs use `validation_split` on the training data, and some models were also fitted on the validation set, so validation scores for those runs are not clean.
- **Limited real-world evidence.** Robustness to night-time scenes, heavy occlusion, other helmet types or camera angles has not been tested.
- **Image-level classification only.** The model classifies one image as helmet or no helmet. It does not locate individual workers, so a crowded frame with mixed compliance is not handled.

## Business Recommendations
- The pre-trained VGG-16 features are very effective for this task, and a frozen backbone with a small head trains quickly on little data.
- Before deployment, validate on a larger, independent set of site images, including hard cases (low light, occlusion, distance).
- Fix the loss-function mismatch in the weaker models and re-run the comparison for a fair benchmark.
- Possible improvements: other transfer-learning backbones (ResNet, EfficientNet, MobileNet for edge devices), stronger augmentation, tuned dropout and fine-tuning of upper VGG layers.
- For multi-worker scenes, move to an object-detection approach (e.g. YOLO) that localises each worker's head.
- Keep a human in the loop for flagged violations until real-world precision and recall are confirmed.

## Getting Started

### Prerequisites
- Python 3.10+
- A GPU is recommended for training VGG-16, but not required.

### Installation
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn jupyter
```

### Data
Place the dataset in a `data/` folder:
```
data/
├── images_proj.npy      # (631, 200, 200, 3) image array
└── Labels_proj.csv      # single 'Label' column (1 = with helmet, 0 = without)
```
The notebook was written for Google Colab and loads data from Google Drive. If you run it locally, remove the `drive.mount(...)` cell and update the two file paths to point to `data/`.

### Run
```bash
jupyter notebook Project_Introduction_to_Computer_Vision_HelmNet.ipynb
```

## Repository Structure
```
.
├── Project_Introduction_to_Computer_Vision_HelmNet.ipynb   # Full analysis and modelling
├── README.md
└── data/                                                   # (not tracked) images and labels
```

## Tech Stack
Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn.

## Author
<Your Name>
