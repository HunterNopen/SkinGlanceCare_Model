# 🧠 SkinGlanceCare Model

<div align="center">

**AI-Powered skin lesion classification prioritizing Clinical-Viability over Benchmark-SOTA**

[Features](#key-features) • [Results](#results) • [Architecture](#architecture) • [Project Structure](#project-structure) • [Limitations](#limitations) • [Milestones](#milestones) • [Limitations](#limitations) • [Future Plan](#future-directions) • [References](#references) • [Disclaimer](#disclaimer) <br>
🤗HuggingFaceDemo: [SkinGlanceCareHFDemo](https://huggingface.co/spaces/HunterNope/SkinGlanceCare) (See the Disclaimer)

</div>

---

## 📋 Overview

SkinGlanceCare - DeepLearning System for dermoscopic skin lesion classification, designed with a **clinical-first approach**. Unlike traditional models that optimize for balanced accuracy, this system prioritizes **cancer recall**, minimizing missed malignancies at the cost of higher false alarm rates.

> *"All models are wrong, but some are useful"* - George Box

The system classifies skin lesions into 8 categories from the ISIC2019 (+OOD) challenge:

| **Cancer:** | MEL | BCC | SCC | AK | 
| --- | --- | --- | --- | --- |
| **Benign:** | **NV** | **BKL** | **DF** | **VASC** |

---

## ✨ <a name="key-features">Key Features<a name="setup"></a>

### 🎯 Clinical-First Design
- **Cancer Recall Priority**: Custom loss function that penalizes missed cancers 4-6× more than false alarms
- **Risk Stratification**: HIGH/MEDIUM/LOW stratification for high-level understanding
- **Uncertainty Estimation**: MC Dropout provides confidence intervals, not just point predictions
- **LLM Recommendations**: Integrated LLM-Api call to serve as a comrehesive guiding tool

### 🧠 Novel Components
- **MaxRecall Loss**: Asymmetric false-negative penalties, differential label smoothing & recall approximator
- **Multi-Component Certainty Score**: Combines base confidence, calibration, decision clarity & prediction stability
- **OOD Detection**: Identifies images outside the training distribution (healthy skin, non-dermoscopic photos)

### ⚡ Production-Ready
- **Quantized Model**: INT8 quantization via NNCF (48 MB → 17.5 MB, 63.5% reduction)
- **OpenVINO Optimization**: CPU-optimized inference for deployment without GPU
- **Modular Architecture**: Clean separation of concerns with Strategy and Factory patterns

---

## 📊 <a name="results">Results</a>

### Primary Metrics (ISIC 2019 Test Set)

| Metric | Value | Notes |
|--------|-------|-------|
| **Cancer Recall** | 96.15% | Primary optimization target |
| **MEL Recall** | 79.94% | 2nd Most Valuable Metric |
| **Cancer F1** | 65.04% | - |
| **Macro F1** | 61.22% | Trade-off for recall |
| **Accuracy** | 64.68% | Not optimized |

### External Validation (PAD-UFES-20, 3-Fold CV)

| Metric | ISIC 2019 (Mean) | PAD-UFES-20 (Mean) | Delta Diff (Δ) |
|--------|-----------|-------------|---|
| Cancer Recall | 94.14% | 81% | -16% |
| Accuracy | 61.12% | 49% | -25% |
| Macro F1 | 58.13% | 45% | -26% |

> Domain-Shift: Pad-Ufes-20 (Smartphone Clinical Images) - ISIC2019 (Dermoscopic Images). However, preserved reasonably high cancer recall with a basic metrics degradation.

### Threshold Analysis

| Threshold | Cancer Recall | Cancer Precision | Cancer F1 |
|-----------|---------------|------------------|-----------|
| 0.25 | 98.3% | 52.7% | 68.6% |
| 0.35 | 97.2% | 55.1% | 70.3% |
| **0.50** | **96.15%** | **59.9%** | **73.5%** |
| 0.60 | 93.8% | 63.2% | 75.5% |

---

## 🏗️ <a name="architecture">Architecture</a>

### Key Design Decisions

| Decision | Choice | Rejected Alternatives | Rationale |
|----------|--------|----------------------|-----------|
| Architecture | Single multi-class | Cascade, Ensemble | No error propagation, simpler deployment |
| Backbone | EfficientNet-B3 | ResNet, ViT, DenseNet | Best accuracy/parameters ratio |
| Loss | MaxRecall (custom) | CE, Focal, Weighted CE | Asymmetric FN penalty, explicit recall term |
| Sampling | Weighted (2× cancer) | SMOTE, Undersampling | Preserves data, improves recall |
| Inference | TTA + MC Dropout | Standard | Uncertainty estimation, better recall |

```
┌─────────────────────────────────────────────────────────────┐
│                    SkinGlanceCare Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  Input Image                                                │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                    │
│  │  Preprocessing                      │                    │
│  │  • Color Constancy (Shades of Gray) │                    │
│  │  • Resize to 300×300                │                    │
│  │  • ImageNet Normalization           │                    │
│  └─────────────────────────────────────┘                    │
│       ↓                                                     │
│  ┌───────────────────────────────────────┐                  │
│  │  EfficientNet-B3 Backbone             │                  │
│  │  (12M parameters, ImageNet pretrained)|                  │
│  └───────────────────────────────────────┘                  │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                    │
│  │  Classification Head                │                    │
│  │  • Dropout (0.3) → FC(512) → SiLU   │                    │
│  │  • BatchNorm → Dropout (0.2)        │                    │
│  │  • FC(8) → Softmax                  │                    │
│  └─────────────────────────────────────┘                    │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                    │
│  │  Post-Processing                    │                    │
│  │  • MC Dropout (10 samples)          │                    │
│  │  • Certainty Scoring                │                    │
│  │  • Risk Stratification              │                    │
│  │  • OOD Detection                    │                    │
│  └─────────────────────────────────────┘                    │
│       ↓                                                     │
│  Output: Risk Level + Probs + Certainty + Recommendation    |
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 <a name="project-structure">Project Structure</a>
```
SkinGlanceCare_Model/
├── 📄 main.py                    # Main script for train/test
├── 📄 __init__.py                # Project preserves Module structure
│
├── 📁 config/                    # Configuration management
│   └── config.py                 # Dataclass-based config
│
├── 📁 data/                      # Data loading & processing
│   ├── dataset.py                # ISIC dataset class
│   ├── datamodule.py             # Lightning DataModule
│   ├── padufes_dataset.py        # PAD-UFES-20 for validation
│   ├── padufes_datamodule.py     # K-fold CV support
│   └── samplers.py               # Weighted sampling
│
├── 📁 models/                    # Model architecture
│   ├── backbone.py               # EfficientNet factory
│   └── classifier.py             # Main Lightning module
│
├── 📁 losses/                    # Loss functions
│   └── max_recall_loss.py        # RecallMax implementation
│
├── 📁 preprocessing/             # Image preprocessing
│   ├── hair_remover.py           # Hair removal facade
│   ├── hair_strategy_removal.py  # DullRazor & Aggressive strategies
│   └── color_constancy.py        # Shades of Gray, Gray World
│
├── 📁 callbacks/                 # Custom callbacks
│   └── csv_logger.py             # Per-epoch metrics logging
│
├── 📁 utils/                     # Utilities
│   ├── metrics.py                # Cancer vs Benign metrics
│   ├── visualization.py          # Confusion matrix, GradCAM
│   └── trainer_factory.py        # Lightning Trainer setup
│
├── 📁 abstract/                  # Design patterns
│   └── null_object.py            # Null Object pattern
│
├── 📁 notebooks_experiments/     # Jupyter notebooks
│   └── merge_ham_isic.ipynb      # Dataset merging
│
└── 📁 data_eda/                      # Data & EDA
    ├── datasets/                     # ISIC, HAM10000, PAD-UFES-20
    ├──ham_10000_load_dataset.ipynb   # EDA HAM10000
    ├──isic_load_dataset.ipynb        # EDA ISIC2019
    └──pad_load_dataset.ipynb         # EDA PAD-UFES-20
```

---

### Dynamic Configuration

All hyperparameters & special function variables tweaked in 1 Config file.
```python
from config import Config, ModelConfig, LossConfig

cfg = Config(
    model=ModelConfig(
        base_model="efficientnet_b3",
        image_size=384,
        dropout_1=0.3,
    ),
    loss=LossConfig(
        fn_multiplier=4.0,
        mel_fn_multiplier=6.0,
        recall_loss_weight=0.3,
    ),
)

...
```

---

## 🎯 <a name="milestones">Milestones</a>

- [x] **v1.0** - Baseline model with standard cross-entropy loss on HAM10000
- [x] **v1.5** - Weighted sampling and class balancing
- [x] **v2.0** - RecallMax Loss implementation. Migration to ISIC2019 dataset
- [x] **v2.1** - MC Dropout uncertainty estimation
- [x] **v2.2** - Multi-component certainty scoring
- [x] **v2.3** - OOD detection system
- [x] **v2.4** - OpenVINO INT8 quantization
- [x] **v2.5** - External validation on PAD-UFES-20
- [x] **v2.6** - Modular refactoring with design patterns

---

## ⚠️ <a name="limitations">Limitations</a>

### Statistical Rigor
- **Single seed training** - used seed=42 during all experiments
- **No ablation studies** - due to time constraints no methodology isolation tests

### Data Constraints
- **Trained on dermoscopic images only** - domain shift to clinical/smartphone photos
- **Limited external validation** - only PAD-UFES-20 tested
- **No clinical validation** - not evaluated by dermatologists

### Technical Debt
- Hair removal preprocessing **decreased** metrics in experiments (not fully investigated)
- MC Dropout unavailable in OpenVINO mode (no uncertainty in fast inference)
- Not explainable enough, GradCAM shows WHERE model looks, not WHY it decides

### Deployment Caveats
- **Not for standalone diagnosis** - screening support tool only
- Cold start delays on HuggingFace Spaces free tier
- No evolution or time data (tracking lesion changes over time)

---

## 🔮 <a name="future-directions">Future Directions</a>

### SHOULD
- [ ] **Data Gather** - integrate clinical (smartphone) images alongside dermoscopic
- [ ] **Stats Validation** - 5-fold CV with different seeds
- [ ] **Ablation Studies** - isolate components and track their contribution directly

### COULD
- [ ] **Multi-Stage Classifier** - joint segmentation + classification
- [ ] **Multimodality** - metadata integration: age, sex, anatomical location (if quality improves)
- [ ] **Calibration Analysis** - reliability diagrams, expected calibration error

### Exploration
- [ ] **Vision Transformers (ViT)** - DeiT-Small or Swin-Tiny comparison
- [ ] **Knowledge distillation** - compress to mobile-friendly model
- [ ] **Federated learning** - privacy-preserving training across institutions

---

## 📚 <a name="references">References</a>

### Datasets
- [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

## ⚕️ <a name="disclaimer">Disclaimer</a>

**This tool is for educational and research purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified dermatologist with any questions regarding skin lesions. The authors assume no liability for any decisions made based on this tool's output.
