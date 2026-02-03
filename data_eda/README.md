# Datasets (scratchpad)

This folder contains datasets used for skin lesion analysis, which is not included in the repository due to size constraints. Download these datasets separately.

## EDA Contents 
- `isic_load_data.py` - EDA script for ISIC 2019 dataset, including data loading, visualization, and basic statistics. Uses Chi-2, Cramer's V and Theils-U for feature analysis.
- `pad_load_data.py` - EDA script for PAD-UFES-20 dataset, including data loading, visualization, and basic statistics. Uses Chi-2, Cramer's V and Theils-U for feature analysis.
- `ham_load_data.py` - EDA script for HAM10000 dataset. Mainly skipped as similar to (part of) ISIC 2019 EDA.

## Gitignore Data Contents 

- `ISIC_2019_Training_GroundTruth.csv` - training labels (multi-class)
- `ISIC_2019_Training_Metadata.csv` - training metadata (patient/lesion info)
- `ISIC_2019_Test_GroundTruth.csv` - test labels (held-out set)
- `ISIC_2019_Test_Metadata.csv` - test metadata (patient/lesion info)
- `PAD_20_Metadata.csv` - PAD-UFES-20 metadata
- `HAM10000_metadata.csv` - HAM10000 metadata
---
- `ISIC_2019_Training_Input/` - ISIC 2019 training images
- `ISIC_2019_Test_Input/` - ISIC 2019 test images
- `PAD-UFES-20/images/` - PAD-UFES-20 images
- `HAM10000_images` - HAM10000 images