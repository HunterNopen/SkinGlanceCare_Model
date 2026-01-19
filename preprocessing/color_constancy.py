import numpy as np


class ShadesOfGrayAlgorithm:

    def __init__(self, power: int = 6):
        self.power = power

    def apply(self, image: np.ndarray) -> np.ndarray:

        img = np.float32(image)
        img_power = np.power(img + 1e-6, self.power)

        rgb_vec = np.power(np.mean(img_power, axis=(0, 1)), 1.0 / self.power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec / (rgb_norm + 1e-6)
        rgb_vec = 1.0 / (rgb_vec * np.sqrt(3) + 1e-6)

        img = np.multiply(img, rgb_vec)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img


class GrayWorldAlgorithm:

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.float32(image)

        avg_r = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_b = np.mean(img[:, :, 2])

        avg = (avg_r + avg_g + avg_b) / 3.0

        scale_r = avg / (avg_r + 1e-6)
        scale_g = avg / (avg_g + 1e-6)
        scale_b = avg / (avg_b + 1e-6)

        img[:, :, 0] *= scale_r
        img[:, :, 1] *= scale_g
        img[:, :, 2] *= scale_b

        img = np.clip(img, 0, 255).astype(np.uint8)

        return img


class ColorConstancyProcessor:

    def __init__(self, algorithm: str = "shades_of_gray", power: int = 6):

        if algorithm == "shades_of_gray":
            self.algorithm = ShadesOfGrayAlgorithm(power=power)
        elif algorithm == "gray_world":
            self.algorithm = GrayWorldAlgorithm()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.algorithm.apply(image)


def shades_of_gray(image: np.ndarray, power: int = 6) -> np.ndarray:

    algorithm = ShadesOfGrayAlgorithm(power=power)

    return algorithm.apply(image)
