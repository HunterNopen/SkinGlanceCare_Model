import numpy as np
import cv2

from .hair_strategy_removal import AggressiveHairRemovalStrategy, DullRazorStrategy


class HairRemover:
    """Facade class for hair removal strategies.
    
    This class provides a unified interface for different hair removal 
    algorithms using the Strategy pattern. It can be configured to use
    either a standard DullRazor approach or a more aggressive multi-scale
    detection method.
    
    Args:
        kernel_size: Size of morphological kernel for DullRazor (default: 17)
        threshold: Threshold value for binary masking in DullRazor (default: 10)
        inpaint_radius: Radius for inpainting operation (default: 5)
        aggressive: If True, uses AggressiveHairRemovalStrategy, 
                   otherwise uses DullRazorStrategy (default: True)
    
    Example:
        >>> remover = HairRemover(aggressive=False, kernel_size=15)
        >>> clean_image = remover(image_with_hair)
    """

    def __init__(
        self,
        kernel_size: int = 17,
        threshold: int = 10,
        inpaint_radius: int = 5,
        aggressive: bool = True
    ):
        if aggressive:
            self.strategy = AggressiveHairRemovalStrategy()
        else:
            self.strategy = DullRazorStrategy(
                kernel_size=kernel_size,
                threshold=threshold,
                inpaint_radius=inpaint_radius
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Remove hair from the input image.
        
        Args:
            image: Input image as numpy array (grayscale or RGB)
            
        Returns:
            Image with hair removed
        """
        return self.strategy.remove(image)

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate a binary mask of detected hair regions.
        
        This method is useful for visualizing detected hair regions or for 
        use in hyperparameter optimization (e.g., Bayesian searches) to 
        evaluate different detection strategies.
        
        Args:
            image: Input image as numpy array (grayscale or RGB)
            
        Returns:
            Binary mask where hair regions are marked as 255 (white)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        return mask
