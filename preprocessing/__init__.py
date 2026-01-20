from .hair_startegy_removal import DullRazorStrategy, AggressiveHairRemovalStrategy
from .hair_remover import HairRemover
from .color_constancy import (
    ColorConstancyProcessor,
    ShadesOfGrayAlgorithm,
    GrayWorldAlgorithm,
    shades_of_gray,
)

__all__ = [
    "HairRemover",
    "HairRemovalStrategy",
    "DullRazorStrategy",
    "AggressiveHairRemovalStrategy",
    "ColorConstancyProcessor",
    "ColorConstancyAlgorithm",
    "ShadesOfGrayAlgorithm",
    "GrayWorldAlgorithm",
    "shades_of_gray",
]
