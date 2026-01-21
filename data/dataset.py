import os
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from torchvision import transforms

from config import Config
from preprocessing import HairRemover, shades_of_gray


class ISICDataset(Dataset):

    def __init__(
        self,
        img_dir: str, dataframe: pd.DataFrame,
        config: Config,transform: A.Compose = None, is_training: bool = True):

        self.img_dir = img_dir
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.is_training = is_training

        self.hair_remover = self._initialize_hair_remover()
        self.image_ids = self.df["image"].tolist()
        self.labels = self._extract_labels()
        self.label_indices = np.argmax(self.labels, axis=1)

    def _initialize_hair_remover(self):
        
        if self.config.data.use_hair_removal:
            return HairRemover(kernel_size=self.config.data.hair_removal_kernel,aggressive=True)
        
        return None

    def _extract_labels(self) -> np.ndarray:
        
        label_columns = list(self.config.model.label_classes)

        return self.df[label_columns].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):

        image = self._load_image(idx)
        image = self._apply_preprocessing(image)
        image = self._apply_transforms(image)

        label = int(self.label_indices[idx])

        return image, label

    def _load_image(self, idx: int) -> np.ndarray:
        
        img_name = f"{self.image_ids[idx]}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            return np.array(img)
        
        except FileNotFoundError:
            return self._create_placeholder_image()

    def _create_placeholder_image(self) -> np.ndarray:
        
        size = self.config.model.image_size
        return np.full((size, size, 3), 128, dtype=np.uint8)

    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        
        if self.hair_remover is not None:
            image = self.hair_remover(image)

        if self.config.data.use_color_constancy:
            image = shades_of_gray(image)

        return image

    def _apply_transforms(self, image: np.ndarray):
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image
