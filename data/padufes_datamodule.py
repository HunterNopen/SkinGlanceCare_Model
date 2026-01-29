import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from config import Config
from .padufes_dataset import PadUfes20Dataset


class PadUfes20DataModule(pl.LightningDataModule):

    def __init__(self, config: Config, fold: int = None, n_splits: int = 5):
        super().__init__()

        self.config = config
        self.fold = fold
        self.n_splits = n_splits

        self.test_transform = self._create_test_transforms()

        self.images_dir = os.path.join(
            config.data.path_train_images.split("ISIC_2019")[0],
            "PAD-UFES-20/images"
        )

        self.metadata_path = os.path.join(
            config.data.path_train_images.split("ISIC_2019")[0],
            "PAD_20_Metadata.csv"
        )

    def _create_test_transforms(self) -> A.Compose:
        
        size = self.config.model.image_size

        return A.Compose([
            A.Resize(int(size * 1.1), int(size * 1.1)),
            A.CenterCrop(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):

        print(f"\nLoading PAD-UFES-20 dataset from {self.metadata_path}...")

        df = pd.read_csv(self.metadata_path)
        df = df[df["diagnostic"].notna()].reset_index(drop=True)

        print(f"Total samples: {len(df)}")
        print(f"Diagnostic distribution:")
        print(df["diagnostic"].value_counts())

        if self.fold is None:
            print("\nMode: External Validation (using all data for testing)")
            self.test_df = df
            self.train_df = None
            self.val_df = None

        else:
            print(f"\nMode: K-Fold CV (fold {self.fold + 1}/{self.n_splits})")

            labels = df["diagnostic"].map(PadUfes20Dataset.CLASS_MAPPING).values
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

            splits = list(skf.split(df, labels))
            train_idx, test_idx = splits[self.fold]

            self.train_df = df.iloc[train_idx].reset_index(drop=True)
            self.test_df = df.iloc[test_idx].reset_index(drop=True)
            self.val_df = None

            print(f"  Train: {len(self.train_df):,}")
            print(f"  Test:  {len(self.test_df):,}")

        self.test_dataset = PadUfes20Dataset(
            self.images_dir,
            self.test_df,
            self.config,
            self.test_transform,
            is_training=False
        )

        if self.train_df is not None:
            self.train_dataset = PadUfes20Dataset(
                self.images_dir,
                self.train_df,
                self.config,
                self.test_transform,
                is_training=False
            )
        else:
            self.train_dataset = None

        self._print_dataset_info()

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers // 2,
            pin_memory=self.config.data.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        
        if self.train_dataset is None:
            raise ValueError("train_dataloader not available in external validation mode")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers // 2,
            pin_memory=self.config.data.pin_memory,
        )

    def _print_dataset_info(self):
        
        print(f"\n{'=' * 60}")
        print(f"PAD-UFES-20 Dataset Ready")
        print(f"Resolution: {self.config.model.image_size}x{self.config.model.image_size}")
        print(f"Hair Removal: {self.config.data.use_hair_removal}")
        print(f"Color Constancy: {self.config.data.use_color_constancy}")
        print(f"{'=' * 60}")

        if self.fold is None:
            print(f"  Test:  {len(self.test_df):,}")
        else:
            print(f"  Train: {len(self.train_df):,}")
            print(f"  Test:  {len(self.test_df):,}")

        print(f"\nClass distribution (Test set):")
        self.test_dataset.get_class_distribution()

        print(f"{'=' * 60}\n")

    @staticmethod
    def create_fold_datamodules(config: Config, n_splits: int = 5):
        return [PadUfes20DataModule(config, fold=i, n_splits=n_splits) for i in range(n_splits)]
