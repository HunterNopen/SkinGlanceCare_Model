import torch
from torch.utils.data import WeightedRandomSampler


class CancerOversamplerFactory:

    @staticmethod
    def create_sampler(
        label_indices: list,
        cancer_classes: set,
        oversample_factor: float = 2.0
    ) -> WeightedRandomSampler:

        weights = []
        for label in label_indices:
            weight = oversample_factor if label in cancer_classes else 1.0
            weights.append(weight)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
