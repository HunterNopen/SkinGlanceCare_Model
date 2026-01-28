import os
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset

from config import Config
from abstract import NullObject
from preprocessing import HairRemover, shades_of_gray

import logging
logger = logging.getLogger(__name__)


class PadUfes20Dataset(Dataset):

    ### Map ISIC idx => PAD-UFES idx
    CLASS_MAPPING = {
        "MEL": 0,
        "NEV": 1,
        "BCC": 2,
        "ACK": 3,
        "SEK": 4,
        "SCC": 7,
    }

    def __init__(self, img_dir: str, dataframe: pd.DataFrame, config: Config, 
                 transform: A.Compose = None, is_training: bool = False):
        
        self.img_dir = img_dir
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.is_training = is_training

        self.hair_remover = self._initialize_hair_remover()


        self.image_ids = self.df["img_id"].tolist()


        self.label_indices = self._map_labels()


        self.labels = self._create_one_hot_labels()

    def _initialize_hair_remover(self):
        
        if self.config.data.use_hair_removal:
            return HairRemover(
                kernel_size=self.config.data.hair_removal_kernel,
                aggressive=False
            )
        return NullObject()

    def _map_labels(self) -> np.ndarray:
        
        label_indices = []

        for diagnostic in self.df["diagnostic"]:
            if diagnostic in self.CLASS_MAPPING:
                label_indices.append(self.CLASS_MAPPING[diagnostic])
            else:
                logger.warning(f"Unknown diagnostic: {diagnostic}, skipping sample")
                label_indices.append(-1)

        return np.array(label_indices, dtype=np.int64)

    def _create_one_hot_labels(self) -> np.ndarray:
        
        num_samples = len(self.label_indices)
        one_hot = np.zeros((num_samples, self.config.model.num_classes), dtype=np.float32)

        for i, label_idx in enumerate(self.label_indices):
            if label_idx >= 0:
                one_hot[i, label_idx] = 1.0

        return one_hot

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        
        image = self._load_image(idx)
        image = self._apply_preprocessing(image)
        image = self._apply_transforms(image)

        label = int(self.label_indices[idx])

        return image, label

    def _load_image(self, idx: int) -> np.ndarray:
        
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")

            return np.array(img)
        
        except FileNotFoundError:
            logger.warning(f"Image not found at {img_path}, using placeholder")

            return self._create_placeholder_image()

    def _create_placeholder_image(self) -> np.ndarray:
        
        size = self.config.model.image_size

        return np.full((size, size, 3), 128, dtype=np.uint8)

    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        
        if self.config.data.use_hair_removal:
            image = self.hair_remover(image)

        if self.config.data.use_color_constancy:
            image = shades_of_gray(image)

        return image

    def _apply_transforms(self, image: np.ndarray):
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)["image"]
            else:
                from torchvision import transforms
                image = transforms.ToTensor()(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        return image

    def get_class_distribution(self):
        
        unique, counts = np.unique(self.label_indices, return_counts=True)
        distribution = dict(zip(unique, counts))

        print("\nPAD-UFES-20 Class Distribution:")
        for idx, count in sorted(distribution.items()):
            if idx >= 0:
                class_name = self.config.model.label_classes[idx]
                is_cancer = "CANCER" if idx in self.config.model.cancer_classes else "BENIGN"
                print(f"  {class_name} (idx {idx}): {count:,} ({is_cancer})")

        return distribution
