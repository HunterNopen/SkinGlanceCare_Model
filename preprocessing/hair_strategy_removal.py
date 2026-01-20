import numpy as np
import cv2
from abc import ABC, abstractmethod


class HairRemovalStrategy(ABC):
    """Abstract base class for hair removal strategies.
    
    All hair removal strategies must implement the remove() method
    which takes an image and returns a processed image with hair removed.
    """
    
    @abstractmethod
    def remove(self, image: np.ndarray) -> np.ndarray:
        """Remove hair from the input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image with hair removed
        """
        pass


class DullRazorStrategy(HairRemovalStrategy):
    """Standard hair removal strategy using morphological operations.
    
    This strategy implements the DullRazor algorithm which uses:
    - Grayscale conversion
    - Morphological blackhat operation to detect dark hair
    - Binary thresholding to create a mask
    - Mask dilation for better coverage
    - Inpainting to fill the masked regions
    
    Best for: Images with moderate hair presence and standard lighting.
    
    Args:
        kernel_size: Size of morphological kernel (default: 17)
        threshold: Threshold value for binary masking (default: 10)
        inpaint_radius: Radius for inpainting operation (default: 5)
    
    Example:
        >>> strategy = DullRazorStrategy(kernel_size=15, threshold=12)
        >>> clean_image = strategy.remove(image_with_hair)
    """

    def __init__(self, kernel_size: int = 17, threshold: int = 10, inpaint_radius: int = 5):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.inpaint_radius = inpaint_radius

    def remove(self, image: np.ndarray) -> np.ndarray:
        """Remove hair using DullRazor algorithm.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Image with hair removed
        """
        image = self._ensure_rgb(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (self.kernel_size, self.kernel_size)
        )

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, mask = cv2.threshold(
            blackhat, self.threshold, 255, cv2.THRESH_BINARY
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

        result = cv2.inpaint(
            image, mask,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA
        )

        return result

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale image to RGB if needed.
        
        Args:
            image: Input image
            
        Returns:
            RGB image
        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image


class AggressiveHairRemovalStrategy(HairRemovalStrategy):
    """Advanced multi-scale hair removal strategy.
    
    This strategy uses a comprehensive approach combining:
    - Multi-scale morphological operations with multiple kernel sizes
    - Both blackhat (dark hair) and tophat (light hair) detection
    - Horizontal and vertical line detection for thin hairs
    - Mask combination and cleanup operations
    - Enhanced inpainting
    
    Best for: Images with heavy hair presence, varying hair thickness,
             or challenging lighting conditions.
    
    Args:
        kernel_sizes: Tuple of kernel sizes for multi-scale detection 
                     (default: (9, 17, 25, 35))
    
    Example:
        >>> strategy = AggressiveHairRemovalStrategy(kernel_sizes=(7, 15, 23))
        >>> clean_image = strategy.remove(heavily_haired_image)
    """

    def __init__(self, kernel_sizes=(9, 17, 25, 35)):
        self.kernel_sizes = kernel_sizes

    def remove(self, image: np.ndarray) -> np.ndarray:
        """Remove hair using aggressive multi-scale approach.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Image with hair removed
        """
        image = self._ensure_rgb(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        masks = self._detect_hair_multiscale(gray)
        masks.extend(self._detect_thin_hair_lines(gray))

        final_mask = self._combine_masks(masks)
        final_mask = self._cleanup_mask(final_mask)

        result = cv2.inpaint(
            image, final_mask,
            inpaintRadius=7,
            flags=cv2.INPAINT_TELEA
        )

        return result

    def _detect_hair_multiscale(self, gray: np.ndarray) -> list:
        """Detect hair at multiple scales using morphological operations.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of binary masks for each scale
        """
        masks = []

        for ksize in self.kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask_dark = cv2.threshold(blackhat, 8, 255, cv2.THRESH_BINARY)

            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            _, mask_light = cv2.threshold(tophat, 8, 255, cv2.THRESH_BINARY)

            masks.append(mask_dark | mask_light)

        return masks

    def _detect_thin_hair_lines(self, gray: np.ndarray) -> list:
        """Detect thin hair lines using directional kernels.
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of masks for horizontal and vertical lines
        """
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        line_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_line_h)
        line_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_line_v)

        _, mask_line_h = cv2.threshold(line_h, 10, 255, cv2.THRESH_BINARY)
        _, mask_line_v = cv2.threshold(line_v, 10, 255, cv2.THRESH_BINARY)

        return [mask_line_h, mask_line_v]

    def _combine_masks(self, masks: list) -> np.ndarray:
        """Combine multiple masks using OR operation.
        
        Args:
            masks: List of binary masks
            
        Returns:
            Combined binary mask
        """
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined | mask
            
        return combined

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up mask using morphological operations.
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned binary mask
        """
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean)
        mask = cv2.dilate(mask, kernel_clean, iterations=2)

        return mask

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale image to RGB if needed.
        
        Args:
            image: Input image
            
        Returns:
            RGB image
        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image