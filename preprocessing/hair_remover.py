import numpy as np
import cv2

from .hair_startegy_removal import AggressiveHairRemovalStrategy, DullRazorStrategy


class HairRemover:
    

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
        
        return self.strategy.remove(image)

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        else: gray = image

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        return mask
