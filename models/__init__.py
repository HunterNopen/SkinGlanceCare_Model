"""Model components for skin lesion classification."""

from .backbone import BackboneFactory, LayerFreezer
from .classifier import SkinLesionClassifier, ClassificationHead

__all__ = [
    "BackboneFactory",
    "LayerFreezer",
    "SkinLesionClassifier",
    "ClassificationHead",
]
