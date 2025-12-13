# Preprocessing

This folder contains preprocessing pipelines and scripts for normalizing skin lesion datasets and preparing them for model training and evaluation.

## Purpose

The preprocessing module handles:
- **Image normalization** — resizing, color correction, and standardization
- **Data augmentation** — generating training variations (rotation, flip, brightness, etc.)
- **Label encoding** — converting multi-class labels to model-ready format
- **Metadata integration** — merging image data with patient/lesion metadata
- **Train/validation/test splits** — creating balanced dataset partitions
- **Artifact removal** — handling hair, rulers, and other image artifacts

## Expected Scripts

- `prepare_isic.py` — pipeline for ISIC 2019 dataset preparation
- `prepare_pad.py` — pipeline for PAD-UFES-20 dataset preparation
- `split_data.py` — create stratified train/val/test splits
- `preprocess_pipeline.py` — unified preprocessing orchestration

## Future (possible) Enhancements

- [ ] Hair removal preprocessing using morphological operations
- [ ] Color constancy algorithms for lighting normalization
- [ ] Automatic artifact detection and cropping
- [ ] Integration with DVC for data versioning

## Notes

- Preprocessed outputs will be saved to `../data/processed/`