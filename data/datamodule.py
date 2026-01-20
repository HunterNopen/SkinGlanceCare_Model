import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config
from .dataset import ISICDataset
from .samplers import CancerOversamplerFactory


class ISICDataModule(pl.LightningDataModule):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.class_counts = None

        self.train_transform = self._create_train_transforms()
        self.val_transform = self._create_val_transforms()

    def _create_train_transforms(self) -> A.Compose:

        size = self.config.model.image_size

        return A.Compose([
            A.RandomResizedCrop(
                size=(size, size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1)
            ),
            A.Rotate(limit=180, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(5, 20),
                hole_width_range=(5, 20),
                fill=0,
                p=0.4
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _create_val_transforms(self) -> A.Compose:
        """Create validation/test transforms."""
        size = self.config.model.image_size

        return A.Compose([
            A.Resize(int(size * 1.1), int(size * 1.1)),
            A.CenterCrop(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        
        print("Loading ISIC 2019 dataset...")

        gt_df = pd.read_csv(self.config.data.path_train_gt)
        meta_df = pd.read_csv(self.config.data.path_train_meta)

        full_df = pd.merge(
            gt_df,
            meta_df[["image", "lesion_id"]],
            on="image",
            how="inner",
        )

        if "UNK" in full_df.columns:
            full_df = full_df[full_df["UNK"] == 0.0]

        if os.path.exists(self.config.data.path_test_gt):
            print("Using separate test set...")
            self.train_df, self.val_df = self._create_train_val_splits(full_df)
            self.test_df = self._load_test_set()
            self.test_images_dir = self.config.data.path_test_images
        else:
            print("Creating train/val/test splits from training data...")
            self.train_df, self.val_df, self.test_df = self._create_three_way_splits(full_df)
            self.test_images_dir = self.config.data.path_train_images

        self.class_counts = self._compute_class_counts()

        self._print_dataset_info()

        self.train_dataset = ISICDataset(
            self.config.data.path_train_images,
            self.train_df,
            self.config,
            self.train_transform,
            is_training=True
        )

        self.val_dataset = ISICDataset(
            self.config.data.path_train_images,
            self.val_df,
            self.config,
            self.val_transform,
            is_training=False
        )

        # NOTE: For debugging or faster experiments, you may temporarily subset self.test_df = self.test_df[self.test_df['UNK'] == 1.0][:100]
        self.test_dataset = ISICDataset(
            self.test_images_dir,
            self.test_df,
            self.config,
            self.val_transform,
            is_training=False
        )

        self.train_sampler = self._create_weighted_sampler()

    def _load_test_set(self) -> pd.DataFrame:
        
        test_gt_df = pd.read_csv(self.config.data.path_test_gt)

        if os.path.exists(self.config.data.path_test_meta):
            test_meta_df = pd.read_csv(self.config.data.path_test_meta)
            test_df = pd.merge(
                test_gt_df,
                test_meta_df["image"],
                on="image",
                how="left",
            )

        else:
            test_df = test_gt_df
            if "lesion_id" not in test_df.columns:
                test_df["lesion_id"] = test_df["image"]

        return test_df

    def _create_train_val_splits(self, df: pd.DataFrame):
        
        print("Performing lesion-level stratified split (train/val)...")

        grouped = df.groupby("lesion_id").first().reset_index()
        labels = np.argmax(
            grouped[list(self.config.model.label_classes)].values,
            axis=1
        )

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(grouped, labels))

        train_lesions = grouped.iloc[train_idx]["lesion_id"].values
        val_lesions = grouped.iloc[val_idx]["lesion_id"].values

        train_set = df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True)
        val_set = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)

        return train_set, val_set

    def _create_three_way_splits(self, df: pd.DataFrame):
        
        print("Performing lesion-level stratified split (train/val/test)...")

        grouped = df.groupby("lesion_id").first().reset_index()
        labels = np.argmax(
            grouped[list(self.config.model.label_classes)].values,
            axis=1
        )

        splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, temp_idx = next(splitter1.split(grouped, labels))

        train_lesions = grouped.iloc[train_idx]["lesion_id"].values
        temp_df = grouped.iloc[temp_idx]
        temp_labels = labels[temp_idx]

        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(splitter2.split(temp_df, temp_labels))

        val_lesions = temp_df.iloc[val_idx]["lesion_id"].values
        test_lesions = temp_df.iloc[test_idx]["lesion_id"].values

        train_set = df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True)
        val_set = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)
        test_set = df[df["lesion_id"].isin(test_lesions)].reset_index(drop=True)

        return train_set, val_set, test_set

    def _compute_class_counts(self) -> np.ndarray:
        
        label_columns = list(self.config.model.label_classes)
        return self.train_df[label_columns].sum().values

    def _create_weighted_sampler(self):
        
        return CancerOversamplerFactory.create_sampler(
            label_indices=self.train_dataset.label_indices,
            cancer_classes=set(self.config.model.cancer_classes),
            oversample_factor=self.config.data.cancer_oversample_factor
        )

    def train_dataloader(self) -> DataLoader:
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=self.train_sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers,
            prefetch_factor=self.config.data.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=self.config.data.pin_memory,
        )

    def _print_dataset_info(self):
        
        print(f"\n{'=' * 60}")
        print(f"Dataset Ready (Resolution {self.config.model.image_size}x{self.config.model.image_size})")
        print(f"Hair Removal: {self.config.data.use_hair_removal}")
        print(f"Color Constancy: {self.config.data.use_color_constancy}")
        print(f"Cancer Oversampling: {self.config.data.cancer_oversample_factor}x")
        print(f"{'=' * 60}")
        print(f"  Train: {len(self.train_df):,}")
        print(f"  Val:   {len(self.val_df):,}")
        print(f"  Test:  {len(self.test_df):,}")

        test_has_ood = "UNK" in self.test_df.columns and self.test_df["UNK"].sum() > 0
        if test_has_ood:
            ood_count = int(self.test_df["UNK"].sum())
            print(f"    *(includes {ood_count} OOD samples)")

        print(f"\nClass distribution (Train):")

        for i, cls in enumerate(self.config.model.label_classes):
            count = int(self.class_counts[i])
            is_cancer = "CANCER" if i in self.config.model.cancer_classes else "BENIGN"
            print(f"  {cls}: {count:,} ({is_cancer})")

        print(f"{'=' * 60}\n")
