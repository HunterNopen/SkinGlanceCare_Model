# Notebooks & Experiments

This folder contains Jupyter notebooks for eda, experiments model and prototyping skin lesion classification approaches.

## Purpose

The notebooks & experiments module handles:
- **Model prototyping** - testing different architectures (CNNs, ViTs, etc)
- **Hyperparameter tuning** - experimenting with learning rates, batch sizes, and optimization strategies
- **Performance benchmarking** - comparing model metrics across different configurations
- **Visualization** - generating plots for loss curves, confusion matrices, and attention maps

## Future (possible) Enhancements

- [ ] Ensemble learning with stacking and pasting
- [ ] Uncertainty quantification and confidence calibration
- [ ] Integration with MLflow for experiment tracking

## Notes

- Experiment results and artifacts will are logged to `../experiments/`
- Implemented "working" code to `../models/` or `../training/` modules