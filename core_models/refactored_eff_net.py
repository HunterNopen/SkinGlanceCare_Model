import torch
import pytorch_lightning as pl
import os
import argparse
import numpy as np
import pandas as pd

from PIL import Image
import seaborn as sns
import torchmetrics
import albumentations as A

from torchvision import transforms, models
from typing import List, Dict
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.model_selection import StratifiedShuffleSplit

torch.cuda.empty_cache()

# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    base_model: str = "efficientnet_b4"
    csv_path: str = os.getenv("CSV_PATH", "./preprocessed_dataset")
    path_images: str = os.getenv("IMAGES_PATH", "./dataset/surajghuwalewala/ham1000-segmentation-and-classification/versions/2/images")
    path_healthy: str = os.getenv("HEALTHY_PATH", "./dataset/MCVSLD/Skin Lesion Dataset/train/Healthy")

    num_classes: int = 8
    label_classes: tuple = ('MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'HEAL')

    batch_size: int = 96
    accumulate_grad_batches: int = 2
    image_size: int = 224
    num_workers: int = 12
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    multiprocessing_context = "spawn"

    max_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    precision: str = "bf16-mixed"

    use_weighted_sampler: bool = False
    use_smote: bool = True
    use_smote_startegy = "proportional"  # "equal"

    cache_in_memory: bool = False
    channels_last: bool = True
    cudnn_benchmark: bool = True


# -----------------------------
# Dataset
# -----------------------------

class HAM10000Dataset(Dataset):
    def __init__(self, img_dirs: List[str], df: pd.DataFrame, transform=None, cache: bool = False):
        self.img_dirs = img_dirs
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache = cache

        self.ids = self.df.iloc[:, 0].astype(str).tolist()
        self.labels = np.argmax(self.df.iloc[:, 1:].values.astype(float), axis=1).astype(int)

        self._cache = [None] * len(self.ids) if cache else None

    def __len__(self):
        return len(self.ids)

    def _get_path(self, img_id: str) -> str:
        name = img_id + ".jpg"
        if name.startswith("ISIC"):
            return os.path.join(self.img_dirs[0], name)
        else:
            return os.path.join(self.img_dirs[1], name)

    def __getitem__(self, idx):
        if self.cache and self._cache[idx] is not None:
            img = self._cache[idx]
        else:
            path = self._get_path(self.ids[idx])
            img = Image.open(path).convert("RGB")

        if self.transform:
            if isinstance(self.transform, A.Compose):
                img = self.transform(image=np.array(img))['image']
            else:
                img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.cache:
            self._cache[idx] = img.clone() if isinstance(img, torch.Tensor) else img

        label = int(self.labels[idx])
        return img, label


# -----------------------------
# DataModule
# -----------------------------

class SkinLesionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.data = pd.read_csv(cfg.csv_path)

        self.train_transform = A.Compose([
            A.RandomResizedCrop(size=(cfg.image_size, cfg.image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),

            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
            ], p=0.7),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
            ], p=0.7),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.GaussNoise(var_limit=(5.0, 20.0), p=1),
            ], p=0.3),

            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(int(cfg.image_size * 1.1), int(cfg.image_size * 1.1)),
            A.CenterCrop(cfg.image_size, cfg.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2(),
        ])

        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_sampler = None

    def setup(self, stage=None):
        self.train_df, self.val_df, self.test_df = self._get_stratified_splits(self.data)

        self.train_dataset = HAM10000Dataset(
            [self.cfg.path_images, self.cfg.path_healthy],
            self.train_df,
            transform=self.train_transform,
            cache=self.cfg.cache_in_memory,
        )

        self.val_dataset = HAM10000Dataset(
            [self.cfg.path_images, self.cfg.path_healthy],
            self.val_df,
            transform=self.val_transform,
            cache=False,
        )

        self.test_dataset = HAM10000Dataset(
            [self.cfg.path_images, self.cfg.path_healthy],
            self.test_df,
            transform=self.val_transform,
            cache=False,
        )

        if self.cfg.use_weighted_sampler:
            print(f"\n{'='*60}\n")
            print("Applying WeightedSampler to preserve the imbalance...")
            labels = np.array(self.train_dataset.labels)
            class_counts = np.bincount(labels, minlength=self.cfg.num_classes).astype(np.float32)
            class_weights = 1.0 / (class_counts + 1e-12)
            sample_weights = class_weights[labels]
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

        if self.cfg.use_smote:
            print(f"\n{'='*60}\n")
            print(f"Applying SMOTE-like oversampling: strategy - '{self.cfg.use_smote_startegy}'...")
            self.train_dataset = self._apply_oversampling(
                self.train_dataset,
                strategy=self.cfg.use_smote_startegy,
                factor=0.15,
                max_limit=1200
            )

        print(f"\n{'='*60}")
        print("Dataset Split Summary:")
        self._print_dataset_summary()
        print(f"{'='*60}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            drop_last=True,
            multiprocessing_context=self.cfg.multiprocessing_context,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=min(len(self.val_dataset), self.cfg.batch_size * 2),
            shuffle=False,
            num_workers=4,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True,
            multiprocessing_context=self.cfg.multiprocessing_context,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=self.cfg.pin_memory,
            multiprocessing_context=self.cfg.multiprocessing_context,
        )

    def _get_stratified_splits(self, df: pd.DataFrame):
        values = df.iloc[:, 0]
        labels = df.iloc[:, 1:]
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(s1.split(values, labels))

        s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        val_idx, test_idx = next(s2.split(df.iloc[temp_idx, 0], df.iloc[temp_idx, 1:]))

        train_set = df.iloc[train_idx].reset_index(drop=True)
        val_set = df.iloc[temp_idx].iloc[val_idx].reset_index(drop=True)
        test_set = df.iloc[temp_idx].iloc[test_idx].reset_index(drop=True)
        return train_set, val_set, test_set

    def _extract_labels(self, dataset):
        if hasattr(dataset, "labels"):
            return np.array(dataset.labels)
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels"):
            return np.array(dataset.dataset.labels)[dataset.indices]
        else:
            raise AttributeError("Cannot find labels attribute in dataset :(")

    def _apply_oversampling(self, dataset, strategy="proportional", factor=0.25, max_limit=1200):
        labels = np.array(dataset.labels)
        class_counts = np.bincount(labels)
        n_classes = len(class_counts)

        print("Class distribution before oversampling:")
        for i, c in enumerate(class_counts):
            print(f"  Class {self.cfg.label_classes[i]}: {c}")

        oversampled_indices = []
        max_count = class_counts.max()
        rng = np.random.default_rng(42)

        for cls in range(n_classes):
            cls_indices = np.nonzero(labels == cls)[0]
            count = len(cls_indices)

            if strategy == "equal":
                target = min(max_count, max_limit)
            elif strategy == "proportional":
                target = int(count + factor * (max_count - count))
            else:
                target = count

            if target > count:
                extra = rng.choice(cls_indices, size=target - count, replace=True)
                oversampled_indices.extend(extra)

        all_indices = np.concatenate([np.arange(len(dataset)), oversampled_indices])
        rng.shuffle(all_indices)

        return Subset(dataset, all_indices)

    def _print_dataset_summary(self):
        for name, dataset in [
            ("Train", self.train_dataset),
            ("Val", self.val_dataset),
            ("Test", self.test_dataset),
        ]:
            labels = self._extract_labels(dataset)
            label_counts = np.bincount(labels, minlength=self.cfg.num_classes)
            print(f" {name}: {len(dataset):,} samples")
            for i, count in enumerate(label_counts):
                label_name = self.cfg.label_classes[i]
                print(f"  {label_name:<15}: {count}")
            print(f"{'=' * 60}\n")


# -----------------------------
# Loss Function
# -----------------------------

class ClinicalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=.0):
        super().__init__()
        self.alpha = alpha.to('cuda') if alpha is not None else None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class AdaptiveClinicalLoss(nn.Module):

    def __init__(self, class_counts=None, gamma=2.0, label_smoothing=0.02, mel_boost=1.5, nv_scale=0.8):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.mel_idx = 0  # MEL
        self.nv_idx = 1   # NV
        self.mel_boost = mel_boost
        self.nv_scale = nv_scale

        weights = self._calculate_adaptive_weights(class_counts)
        self.register_buffer("alpha", weights)

    def _calculate_adaptive_weights(self, class_counts):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num

        weights = weights / weights.sum() * len(weights)

        weights[self.mel_idx] *= self.mel_boost
        weights[self.nv_idx] *= self.nv_scale

        weights = weights / weights.sum() * len(weights)
        return weights

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# -----------------------------
# Lightning Module
# -----------------------------

class SkinGlanceCareClassifier(pl.LightningModule):
    def __init__(self, cfg: Config, class_counts=None):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg

        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self._freeze_early_layers()

        in_feats = self.backbone.classifier[1].in_features

        self.attention = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.Tanh(),
            nn.Linear(512, in_feats),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.num_classes)
        )

        self.loss_fn = AdaptiveClinicalLoss(
            class_counts=class_counts,
            gamma=2.5,
            label_smoothing=0.02,
            mel_boost=1.5,
            nv_scale=0.8
        )

        self._setup_metrics()
        self.sample_images: Dict[int, Dict] = {}

    def on_fit_start(self):
        if self.cfg.channels_last:
            self.to(memory_format=torch.channels_last)

    def forward(self, x):
        if self.cfg.channels_last and x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)

        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)

        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        return self.classifier(attended_features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        f1 = self.train_f1.compute()
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        val_preds = torch.cat(self.val_preds)
        val_labels = torch.cat(self.val_labels)

        if (self.current_epoch % 5 == 4 or self.current_epoch == 0) and not self.trainer.sanity_checking:
            cm = confusion_matrix(val_labels.numpy(), val_preds.numpy())
            self._plot_confusion_matrix(cm, "Validation")

            precision = self.val_precision.compute().cpu().numpy()
            recall = self.val_recall.compute().cpu().numpy()
            self._log_per_class_metrics(precision, recall)

        mel_f1, mel_prec, mel_rec = self._compute_binary_metrics_mel_vs_rest(val_preds, val_labels)
        can_f1, can_prec, can_rec = self._compute_binary_metrics_cancer_vs_benign(val_preds, val_labels)

        self.log('val_mel_vs_rest_f1', mel_f1, prog_bar=True)
        self.log('val_mel_vs_rest_precision', mel_prec)
        self.log('val_mel_vs_rest_recall', mel_rec, prog_bar=True)

        self.log('val_cancer_vs_benign_f1', can_f1, prog_bar=True)
        self.log('val_cancer_vs_benign_precision', can_prec)
        self.log('val_cancer_vs_benign_recall', can_rec, prog_bar=True)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())

        if batch_idx % 20 == 5:
            for i, lbl in enumerate(y):
                cls = int(lbl.item())
                if cls not in self.sample_images:
                    self.sample_images[cls] = {
                        "image": x[i].detach().cpu().clone(),
                        "label": cls,
                        "pred": int(preds[i].item()),
                    }

        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_precision.update(preds, y)
        self.test_recall.update(preds, y)

        self.log('test_loss', loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        acc = self.test_acc.compute()
        f1 = self.test_f1.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()

        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)

        test_preds = torch.cat(self.test_preds)
        test_labels = torch.cat(self.test_labels)
        cm = confusion_matrix(test_labels.numpy(), test_preds.numpy())
        self._plot_confusion_matrix(cm, "Test")

        mel_f1, mel_prec, mel_rec = self._compute_binary_metrics_mel_vs_rest(test_preds, test_labels)
        can_f1, can_prec, can_rec = self._compute_binary_metrics_cancer_vs_benign(test_preds, test_labels)

        print("\n" + "="*80)
        print("Test Results - Per-Class Metrics:")
        print("="*80)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'Instances':<10}")
        print("-"*80)
        for i, cls_name in enumerate(self.cfg.label_classes):
            support = (test_labels == i).sum().item()
            print(f"{cls_name:<10} {precision[i]:.4f} {recall[i]:.4f} {support:<10}")
        print("-"*80)
        print(f"{'Overall':<10} {'Acc: ' + f'{acc:.4f}':<12} {'F1: ' + f'{f1:.4f}':<12}")
        print("-"*80)
        print(f"{'MEL vs rest':<10} F1: {mel_f1:.4f}  Prec: {mel_prec:.4f}  Rec: {mel_rec:.4f}")
        print(f"{'Cancer vs ben.':<10} F1: {can_f1:.4f}  Prec: {can_prec:.4f}  Rec: {can_rec:.4f}")
        print("="*80 + "\n")

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_preds.clear()
        self.test_labels.clear()

    def _compute_binary_metrics_mel_vs_rest(self, preds, labels):
        mel_idx = 0
        binary_labels = (labels == mel_idx).long()
        binary_preds = (preds == mel_idx).long()

        tp = ((binary_preds == 1) & (binary_labels == 1)).sum().float()
        fp = ((binary_preds == 1) & (binary_labels == 0)).sum().float()
        fn = ((binary_preds == 0) & (binary_labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall

    def _compute_binary_metrics_cancer_vs_benign(self, preds, labels):
        cancer_classes = {0, 2, 3}
        binary_labels = torch.tensor([1 if int(l.item()) in cancer_classes else 0 for l in labels])
        binary_preds = torch.tensor([1 if int(p.item()) in cancer_classes else 0 for p in preds])

        tp = ((binary_preds == 1) & (binary_labels == 1)).sum().float()
        fp = ((binary_preds == 1) & (binary_labels == 0)).sum().float()
        fn = ((binary_preds == 0) & (binary_labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall

    def _plot_confusion_matrix(self, cm: np.ndarray, title: str = "Validation"):
        cmn = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cmn,
            annot=True,
            fmt='.2f',
            cmap="Blues",
            ax=ax,
            xticklabels=self.cfg.label_classes,
            yticklabels=self.cfg.label_classes,
            cbar_kws={'label': 'Normalized Count'}
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"{title} Confusion Matrix (Epoch {self.current_epoch})", fontsize=14)
        plt.tight_layout()
        self.logger.experiment.add_figure(
            f"{title}_Confusion_Matrix",
            fig,
            self.current_epoch
        )
        plt.close(fig)

    def _log_per_class_metrics(self, precision: np.ndarray, recall: np.ndarray):
        for i, cls_name in enumerate(self.cfg.label_classes):
            self.logger.experiment.add_scalars(
                f"PerClass/{cls_name}",
                {
                    "precision": precision[i],
                    "recall": recall[i],
                },
                self.current_epoch,
            )

    def _find_last_conv_module(self, module: nn.Module):
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _generate_gradcam_visualizations(self):
        if not self.sample_images:
            print("No sample images for Grad-CAM")
            return

        target_conv = self._find_last_conv_module(self.backbone.features)
        if target_conv is None:
            raise RuntimeError("Not found last conv layer :(")
        target_layers = [target_conv]

        print(f"Target layer for Grad-CAM: {target_conv}")
        cam = GradCAM(model=self, target_layers=target_layers)

        self.eval()
        orig_requires = [p.requires_grad for p in self.parameters()]
        for p in self.parameters():
            p.requires_grad_(True)

        fig, axes = plt.subplots(2, self.cfg.num_classes, figsize=(24, 8))
        try:
            for cls_idx in range(self.cfg.num_classes):
                if cls_idx not in self.sample_images:
                    axes[0, cls_idx].axis('off')
                    axes[1, cls_idx].axis('off')
                    continue

                sample = self.sample_images[cls_idx]
                img_tensor = sample["image"].unsqueeze(0).to(self.device).float()
                true_label = int(sample["label"])
                pred_label = int(sample["pred"])

                targets = [ClassifierOutputTarget(pred_label)]
                with torch.enable_grad():
                    img_tensor.requires_grad_(True)
                    out = self(img_tensor)
                    score = out[0, pred_label]
                    score.backward(retain_graph=True)

                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]

                img_np = img_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)

                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                axes[0, cls_idx].imshow(img_np)
                axes[0, cls_idx].set_title(
                    f"{self.cfg.label_classes[cls_idx]}\nTrue: {self.cfg.label_classes[true_label]}",
                    fontsize=10
                )
                axes[0, cls_idx].axis('off')

                axes[1, cls_idx].imshow(visualization)
                axes[1, cls_idx].set_title(
                    f"Pred: {self.cfg.label_classes[pred_label]}",
                    fontsize=10,
                    color='green' if true_label == pred_label else 'red'
                )
                axes[1, cls_idx].axis('off')

            plt.suptitle("Grad-CAM Visualizations - Model Focus Areas", fontsize=16, y=1.02)
            plt.tight_layout()
            self.logger.experiment.add_figure("GradCAM_Visualizations", fig, self.current_epoch)
            print("Grad-CAM visualizations generated successfully")
        finally:
            plt.close(fig)
            for p, orig in zip(self.parameters(), orig_requires):
                p.requires_grad_(orig)
            try:
                del cam
            except Exception:
                pass

    def _freeze_early_layers(self):
        num_blocks_to_freeze = 3
        for idx, block in enumerate(self.backbone.features):
            if idx < num_blocks_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False

    def _setup_metrics(self):
        num_classes = self.cfg.num_classes

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')

        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average=None)
        self.val_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average=None)

        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average=None)
        self.test_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average=None)

        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []


# -----------------------------
# Utilities
# -----------------------------

def create_trainer(cfg: Config) -> pl.Trainer:
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=2,
            filename='{epoch:02d}-{val_acc:.3f}-{val_f1:.3f}',
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=15,
            mode='max',
            min_delta=0.001,
            verbose=True
        ),
    ]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        logger=TensorBoardLogger('lightning_logs', name='skin_glance_care_classifier_ham10000'),
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_model_summary=True,
        benchmark=True,
        deterministic=False,
    )
    return trainer


def run_training_conf(cfg: Config, datamodule: pl.LightningDataModule, trainer: pl.Trainer) -> pl.LightningModule:
    print("\n" + "="*80)
    print(" " * 20 + "SETTING UP TRAINING PIPELINE!")
    print("="*80 + "\n")

    print("Initializing model...")
    class_counts = datamodule.__getattribute__("data").iloc[:, 1:].sum(axis=0).values
    model = SkinGlanceCareClassifier(cfg, class_counts=class_counts)

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    trainer.fit(model, datamodule=datamodule)

    best_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_checkpoint = callback.best_model_path
            best_score = callback.best_model_score
            print("\n" + "="*80)
            print(f"Best checkpoint: {best_checkpoint}")
            print(f"Best val_f1 score: {best_score:.4f}")
            print("="*80 + "\n")
            break

    return model, best_checkpoint


# -----------------------------
# Main
# -----------------------------

def main(train_pipeline: bool = False, best_checkpoint: str = None):
    print("\n" + "="*80)
    print(" " * 20 + "SKIN GLANCE CARE")
    print("="*80 + "\n")

    cfg = Config()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        if torch.cuda.is_bf16_supported():
            print("Precision - BF16")
        else:
            cfg.precision = "16-mixed"
            print("Precision - FP16")
    else:
        print("WARNING: No GPU detected! Slow training!")
        cfg.precision = "32"

    print("\nConfiguration:")
    print(f"  Model: {cfg.base_model}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Gradient accumulation: {cfg.accumulate_grad_batches}")
    print(f"  Effective batch size: {cfg.batch_size * cfg.accumulate_grad_batches}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max epochs: {cfg.max_epochs}")
    print(f"  Image size: {cfg.image_size}x{cfg.image_size}")
    print(f"  Workers: {cfg.num_workers}")

    print("\nData module setup...")
    datamodule = SkinLesionDataModule(cfg)

    print("Creating trainer...")
    trainer = create_trainer(cfg)

    if train_pipeline:
        model, best_checkpoint = run_training_conf(cfg=cfg, datamodule=datamodule, trainer=trainer)

        if best_checkpoint and os.path.exists(best_checkpoint):
            print("\n" + "="*80)
            print("Loading CHECKPOINTED best model for testing!")
            print("="*80 + "\n")

            best_model = SkinGlanceCareClassifier.load_from_checkpoint(
                best_checkpoint,
                cfg=cfg
            )
            best_model.eval()

            trainer.test(best_model, datamodule=datamodule)

            print("\n" + "="*80)
            print("Generating Grad-CAM visualizations")
            print("="*80 + "\n")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_model.to(device)
            best_model.eval()
            with torch.enable_grad():
                best_model._generate_gradcam_visualizations()
        else:
            print("\n" + "="*80)
            print("Loading TRAINER final model for testing!")
            print("="*80 + "\n")
            trainer.test(model, datamodule=datamodule)
    else:
        assert best_checkpoint is not None, "best_checkpoint path must be provided when skipping training"
        print("\n" + "="*80)
        print("Loading provided checkpoint for testing!")
        print("="*80 + "\n")
        best_model = SkinGlanceCareClassifier.load_from_checkpoint(
            best_checkpoint,
            cfg=cfg
        )
        trainer.test(best_model, datamodule=datamodule)

    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pipeline", action="store_true", default=True)
    parser.add_argument("-s", "--skip_train_pipeline", action="store_false", dest="train_pipeline")
    parser.add_argument(
        "--best_checkpoint",
        type=str,
        default="lightning_logs/skin_glance_care_classifier_ham10000/version_0/checkpoints/epoch=33-val_acc=0.861-val_f1=0.831.ckpt"
    )
    args = parser.parse_args()
    main(**vars(args))