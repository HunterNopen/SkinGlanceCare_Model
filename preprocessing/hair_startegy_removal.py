import numpy as np
import cv2

class DullRazorStrategy:

    def __init__(self, kernel_size: int = 17, threshold: int = 10, inpaint_radius: int = 5):

        self.kernel_size = kernel_size
        self.threshold = threshold
        self.inpaint_radius = inpaint_radius

    def remove(self, image: np.ndarray) -> np.ndarray:
        
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
        
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image


class AggressiveHairRemovalStrategy:

    def __init__(self, kernel_sizes=(9, 17, 25, 35)):
        self.kernel_sizes = kernel_sizes

    def remove(self, image: np.ndarray) -> np.ndarray:

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
        
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        line_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_line_h)
        line_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_line_v)

        _, mask_line_h = cv2.threshold(line_h, 10, 255, cv2.THRESH_BINARY)
        _, mask_line_v = cv2.threshold(line_v, 10, 255, cv2.THRESH_BINARY)

        return [mask_line_h, mask_line_v]

    def _combine_masks(self, masks: list) -> np.ndarray:
        
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined | mask
            
        return combined

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean)
        mask = cv2.dilate(mask, kernel_clean, iterations=2)

        return mask

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image