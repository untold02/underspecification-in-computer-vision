# A Framework for Identifying Underspecification in Image Classification Pipeline Using Post-Hoc Analyzer

[![Conference](https://img.shields.io/badge/ICPRAM-2025-blue)](https://icpram.scitevents.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange.svg)](https://www.tensorflow.org/)

Official implementation of the paper: **"A Framework for Identifying Underspecification in Image Classification Pipeline Using Post-Hoc Analyzer"** by Prabhat Parajuli and Teeradaj Racharak, published at ICPRAM 2025.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

**Underspecification** is a critical challenge in machine learning where models achieve high accuracy during development but fail to generalize to unseen data. This occurs when multiple models with similar performance rely on different features—some relevant, others spurious.

This project introduces a **systematic framework** to identify and quantify underspecification in image classification pipelines by analyzing feature reliance through explainable AI (XAI) techniques.

### The Problem

Models with identical accuracy can learn fundamentally different decision-making strategies:

- **Model A** learns robust features (e.g., object shape, texture)
- **Model B** learns spurious correlations (e.g., background, noise patterns)

Both may perform well on in-distribution data, but **Model B** will fail on new, unseen data. This is the **Rashomon Effect** in machine learning.

### Our Solution

We propose a framework that:

1. **Systematically varies** pipeline components (feature extractors, optimizers, weight initialization)
2. **Generates explanations** using LIME and SHAP for each model variant
3. **Quantifies underspecification** by measuring explanation consistency across models
4. **Identifies** which components contribute most to feature reliance variability

---

## Key Contributions

1. **Novel Framework**: First systematic approach to identify underspecification by varying individual ML pipeline components
2. **XAI-based Detection**: Leverages LIME and SHAP to quantify feature reliance differences across models with similar accuracy
3. **Component-Level Analysis**: Reveals that **feature extractor architecture** is the primary source of underspecification, contributing more than optimizers or weight initialization
4. **Practical Metrics**: Introduces instance-level and class-level underspecification scores for comprehensive analysis
5. **Empirical Validation**: Experiments on three datasets (MNIST, Imagenette, Cats_vs_Dogs) with multiple architectures demonstrate the framework's effectiveness

---

## Methodology

### Framework Overview

```
┌─────────────────────────────────────────────────────────┐
│  ML Pipeline: P = (p₁, p₂, ..., pₙ)                    │
│  Components: Feature Extractor, Optimizer, Init Weights │
└─────────────────────────────────────────────────────────┘
                          ↓
         ┌────────────────────────────────┐
         │  Vary One Component at a Time  │
         │  - Feature Extractors (CNN,    │
         │    MobileNet, DenseNet, etc.)  │
         │  - Optimizers (Adam, SGD, etc.)│
         │  - Weight Initialization Seeds │
         └────────────────────────────────┘
                          ↓
         ┌────────────────────────────────┐
         │   Train Models & Generate      │
         │   Explanations (LIME + SHAP)   │
         └────────────────────────────────┘
                          ↓
         ┌────────────────────────────────┐
         │  Measure Explanation           │
         │  Consistency using Cosine      │
         │  Distance                      │
         └────────────────────────────────┘
                          ↓
         ┌────────────────────────────────┐
         │  Underspecification Score:     │
         │  Higher = More Inconsistent    │
         └────────────────────────────────┘
```

### Underspecification Metrics

**Instance-Level**: Measures explanation consistency for individual test samples

```
d(ξᵐⁱ, ξᵐʲ) := 1 - (ξᵐⁱ · ξᵐʲ) / (||ξᵐⁱ|| ||ξᵐʲ||)
```

**Class-Level**: Aggregates instance-level scores across all samples in a class

```
d̄(m, mⱼ) := (1/|X|) Σ d(ξˣᵐⁱ, ξˣᵐʲ)
```

Where:

- `ξᵐⁱ` = explanation from model mᵢ
- Lower scores = consistent feature reliance (good)
- Higher scores = divergent features (underspecification)

---

## Repository Structure

```
underspecification-in-computer-vision/
│
├── README.md                      # This file
├── paper.pdf                      # Published conference paper
├── requirements.txt               # Python dependencies
├── .gitignore
│
├── mnist/                         # MNIST experiments
│   ├── train1.ipynb              # Training with varying feature extractors
│   ├── train2.ipynb              # Training with varying optimizers/weights
│   ├── generate_lime_expls.ipynb # LIME explanation generation
│   ├── generate_shap_expls.ipynb # SHAP explanation generation
│   ├── instance_level.ipynb      # Instance-level analysis
│   ├── class_level.ipynb         # Class-level analysis
│   ├── tools.py                  # LIME & SHAP explainer classes
│   └── helper_functions.py       # Utility functions
│
├── imagenette/                    # Imagenette experiments
│   ├── train1.ipynb
│   ├── train2.ipynb
│   ├── generate_lime_expls.ipynb
│   ├── generate_shap_expls.ipynb
│   ├── instance_level.ipynb
│   ├── class_level.ipynb
│   ├── tools.py
│   ├── helper_functions.py
│   └── labels.py                 # ImageNet class labels
│
└── cats_vs_dogs/                  # Cats vs Dogs experiments
    ├── train1.ipynb
    ├── train2.ipynb
    ├── generate_lime_expls.ipynb
    ├── generate_shap_expls.ipynb
    ├── instance_level.ipynb
    ├── class_level.ipynb
    ├── tools.py
    └── helper.py
```

### File Descriptions

- **`train1.ipynb`**: Train models with varying feature extractors
- **`train2.ipynb`**: Train models with varying optimizers and weight initialization
- **`generate_lime_expls.ipynb`**: Generate LIME explanations for trained models
- **`generate_shap_expls.ipynb`**: Generate SHAP explanations for trained models
- **`instance_level.ipynb`**: Analyze instance-level underspecification
- **`class_level.ipynb`**: Analyze class-level underspecification
- **`tools.py`**: Core `LIMEExplainer` and `SHAPExplainer` classes
- **`helper_functions.py`**: Data loading, preprocessing, and utility functions

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/prabhat-parajuli/underspecification-in-computer-vision.git
cd underspecification-in-computer-vision
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Dependencies

```
lime==0.2.0.1          # Local Interpretable Model-agnostic Explanations
numpy==2.2.2           # Numerical computing
shap==0.46.0           # SHapley Additive exPlanations
scikit-image           # Image processing
tensorflow==2.14.0     # Deep learning framework
matplotlib==3.8.0      # Visualization
```

---

## Datasets

The framework is evaluated on three image classification datasets with varying complexity:

### 1. **MNIST**

- **Task**: Handwritten digit recognition (0-9)
- **Size**: 60,000 training + 10,000 test images
- **Resolution**: 28×28 grayscale (resized to 32×32×3)
- **Complexity**: Low (simple, clean dataset)
- **Download**: Auto-downloaded via TensorFlow/Keras

### 2. **Imagenette**

- **Task**: 10-class subset of ImageNet
- **Classes**: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute
- **Size**: ~9,500 training + ~3,920 validation images
- **Resolution**: 320×320 (resized to 224×224)
- **Complexity**: High (real-world, diverse)
- **Download**: [fastai/imagenette](https://github.com/fastai/imagenette)

### 3. **Cats vs Dogs**

- **Task**: Binary classification (cat vs dog)
- **Size**: ~25,000 images (from Kaggle competition)
- **Resolution**: Variable (preprocessed and resized)
- **Complexity**: Medium (intra-class variation)
- **Download**: [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats)
- **Note**: Dataset was preprocessed; corrupted images excluded

---

## Usage

### Step 1: Train Models

Navigate to the dataset folder and run the training notebooks:

```bash
cd mnist/
jupyter notebook train1.ipynb  # Train with different feature extractors
jupyter notebook train2.ipynb  # Train with different optimizers/seeds
```

**Training configurations tested:**

- **Feature Extractors**: Custom CNN, MobileNet, DenseNet121, EfficientNetB0, Xception, InceptionV3, ResNet50V2
- **Optimizers**: Adam, SGD, RMSprop, Nadam
- **Weight Initialization**: 10 different random seeds

### Step 2: Generate Explanations

After training, generate LIME and SHAP explanations:

```bash
jupyter notebook generate_lime_expls.ipynb
jupyter notebook generate_shap_expls.ipynb
```

**LIME Configuration:**

- Perturbations: 1,000 samples per instance
- Segmentation: SLIC algorithm (kernel size = 3)
- Top features: 5 (for binary classification: 2)

**SHAP Configuration:**

- Masker: Inpainting (Telea algorithm)
- Max evaluations: 1,000
- Focus: Top-1 predicted class

### Step 3: Analyze Underspecification

Run analysis notebooks to compute underspecification scores:

```bash
jupyter notebook instance_level.ipynb  # Per-instance analysis
jupyter notebook class_level.ipynb     # Per-class aggregated analysis
```

### Example Code Snippet

```python
from tools import LIMEExplainer, SHAPExplainer
import tensorflow as tf

# Load trained models
model1 = tf.keras.models.load_model('models/cnn_model.h5')
model2 = tf.keras.models.load_model('models/mobilenet_model.h5')

# Initialize explainers
lime_explainer = LIMEExplainer()
shap_explainer = SHAPExplainer()

# Generate explanations for a test image
test_image, test_label = test_dataset[0]

lime_expl1 = lime_explainer.explain_aninstance(test_image, model1, num_samples=1000)
lime_expl2 = lime_explainer.explain_aninstance(test_image, model2, num_samples=1000)

# Compute cosine distance between explanations
distance = compute_cosine_distance(lime_expl1, lime_expl2)
print(f"Underspecification Score: {distance:.4f}")
```

---

## Experimental Results

### Key Findings

| Dataset                | Highest Underspecification Component | Avg. ClassLevelScore |
| ---------------------- | ------------------------------------ | -------------------- |
| **MNIST**        | Feature Extractor (0.50)             | 0.45                 |
| **Imagenette**   | Feature Extractor (0.52)             | 0.48                 |
| **Cats_vs_Dogs** | Feature Extractor (0.42)             | 0.38                 |

### Main Insights

1. **Feature extractors dominate underspecification**
   Changing the feature extractor architecture (e.g., CNN → MobileNet) causes the highest variability in learned features.
2. **Optimizers have moderate impact**
   Adam vs. SGD shows some difference, but less than architecture changes.
3. **Weight initialization has minimal impact**
   Different random seeds produce relatively consistent feature reliance (though still measurable).
4. **LIME vs. SHAP consistency**
   LIME explanations show slightly higher variability than SHAP, suggesting sensitivity to perturbation randomness.

### Model Accuracy vs. Underspecification

All models achieved **>97% test accuracy**, yet exhibited significant explanation inconsistency—demonstrating that **high accuracy does not guarantee reliability**.

---

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{parajuli2025framework,
  title     = {A Framework for Identifying Underspecification in Image Classification Pipeline Using Post-Hoc Analyzer},
  author    = {Parajuli, Prabhat and Racharak, Teeradaj},
  booktitle = {Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods (ICPRAM 2025)},
  pages     = {426--436},
  year      = {2025},
  publisher = {SciTePress},
  doi       = {10.5220/0013745203905},
  isbn      = {978-989-758-729-6, 2184-4313}
}
```

**Paper**: [Available here](https://www.scitepress.org/Papers/2025/133742/133742.pdf)
**Conference**: ICPRAM 2025 - 14th International Conference on Pattern Recognition Applications and Methods

---

## License

This project is licensed under the **CC BY-NC 4.0** License - see below for details.

Copyright © 2025 by Scitepress – Science and Technology Publications, Lda.
All rights reserved.

For non-commercial use, distribution, and reproduction, proper attribution must be given.

---

## Acknowledgments

This work was conducted at the **School of Information Science, Japan Advanced Institute of Science and Technology (JAIST)**, Japan.

Special thanks to:

- ICPRAM 2025 reviewers for valuable feedback
- TensorFlow and scikit-learn communities
- LIME and SHAP library maintainers

---

## 👤 Contact

**Prabhat Parajuli**
School of Information Science
Japan Advanced Institute of Science and Technology (JAIST)

For questions or collaboration:

- 📧 Email: [prabhat.p.parajuli@gmail.com](mailto:prabhat.p.parajuli@gmail.com) (primary), [prabhat.dr02@gmail.com](mailto:prabhat.dr02@gmail.com)
- 🔗 LinkedIn: [www.linkedin.com/in/prabhat-parajuli-ba5026229](www.linkedin.com/in/prabhat-parajuli-ba5026229)
- 🐙 GitHub: [https://github.com/prabhat-parajuli/underspecification-in-computer-vision](https://github.com/prabhat-parajuli/underspecification-in-computer-vision)

**Advisor**: Teeradaj Racharak
📧 [r.teeradaj@gmail.com](mailto:r.teeradaj@gmail.com)

---

## 🔗 Related Work

- **Rashomon Effect**: [Breiman, 2001](https://projecteuclid.org/euclid.ss/1009213726)
- **LIME**: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)
- **SHAP**: [Lundberg &amp; Lee, 2017](https://arxiv.org/abs/1705.07874)
- **Underspecification in ML**: [D&#39;Amour et al., 2022](https://arxiv.org/abs/2011.03395)

---

## 🚧 Future Work

- Extension to **vision-language models** (CLIP, BLIP)
- **Automated pipeline recommendation** to minimize underspecification
- **Certified robustness metrics** for safety-critical applications
- Integration with **model cards** for responsible AI deployment

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository! ⭐**

Made with ❤️ for more reliable and interpretable AI systems

</div>
