import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import Config
from .metrics import MetricsCalculator


class _SingleTensorWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get('logits', next(iter(out.values())))
        return out


class TestResultsVisualizer:

    def __init__(self, config: Config):
        self.config = config

    def print_test_summary(
        self,
        acc: float, f1: float, prec_per_class: np.ndarray, rec_per_class: np.ndarray,
        labels: torch.Tensor, preds: torch.Tensor, probs: torch.Tensor, uncertainties: torch.Tensor):
        
        print("\n" + "=" * 70)
        print("TEST RESULTS - FINAL PRODUCTION MODEL")
        print("=" * 70)

        print(f"\nOverall Accuracy: {acc:.4f}")
        print(f"Overall Macro F1: {f1:.4f}")

        print("\nPer-Class Metrics:")
        print("-" * 70)
        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'Support':<10} {'Type'}")
        print("-" * 70)

        for i, name in enumerate(self.config.model.label_classes):
            support = (labels == i).sum().item()
            cls_type = "CANCER" if i in self.config.model.cancer_classes else "BENIGN"
            print(f"{name:<8} {prec_per_class[i]:<12.4f} {rec_per_class[i]:<12.4f} {support:<10} {cls_type}")

        print("-" * 70)

        mel_f1, mel_prec, mel_rec = MetricsCalculator.melanoma_vs_rest_metrics(preds, labels)
        can_f1, can_prec, can_rec = MetricsCalculator.cancer_vs_benign_metrics(
            preds, labels, self.config.model.cancer_classes
        )

        print(f"\n[STANDARD PREDICTIONS]")
        print(f"  MEL:    F1={mel_f1:.4f}  Prec={mel_prec:.4f}  Recall={mel_rec:.4f}")
        print(f"  Cancer: F1={can_f1:.4f}  Prec={can_prec:.4f}  Recall={can_rec:.4f}")

        print("\n" + "-" * 70)
        print("Threshold Analysis:")
        MetricsCalculator.analyze_cancer_thresholds(
            probs, labels, self.config.model.cancer_classes
        )

        print("\n" + "-" * 70)
        print("MC Dropout Uncertainty Analysis:")
        self._analyze_uncertainty(probs, uncertainties, labels)

        print("=" * 70 + "\n")

    def _analyze_uncertainty(self, probs: torch.Tensor, uncertainties: torch.Tensor, labels: torch.Tensor):
        
        max_probs = probs.max(dim=1).values

        high_uncertainty_mask = uncertainties > self.config.inference.uncertainty_threshold
        low_confidence_mask = max_probs < self.config.inference.ood_prob_threshold
        potential_ood = high_uncertainty_mask | low_confidence_mask

        print(f"  Mean uncertainty: {uncertainties.mean():.4f}")
        print(f"  Std uncertainty: {uncertainties.std():.4f}")
        print(f"  High uncertainty samples (>{self.config.inference.uncertainty_threshold}): {high_uncertainty_mask.sum().item()}")
        print(f"  Low confidence samples (<{self.config.inference.ood_prob_threshold}): {low_confidence_mask.sum().item()}")
        print(f"  Potential OOD candidates: {potential_ood.sum().item()}")

        print("\n  Uncertainty by class:")
        for i, cls in enumerate(self.config.model.label_classes):
            cls_mask = labels == i
            if cls_mask.sum() > 0:
                cls_uncertainty = uncertainties[cls_mask].mean()
                cls_confidence = max_probs[cls_mask].mean()
                print(f"    {cls}: uncertainty={cls_uncertainty:.4f}, confidence={cls_confidence:.4f}")

    def plot_confusion_matrix(self, preds: np.ndarray, labels: np.ndarray, output_dir: str, suffix: str = ""):

        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.config.model.label_classes,
            yticklabels=self.config.model.label_classes,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix ({suffix})', fontsize=14)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'confusion_matrix_{suffix}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix: {save_path}")

    def generate_gradcam(self, model, sample_images: dict):
        
        if not sample_images:
            print("No sample images for GradCAM...")
            return

        try:
            target_conv = self._find_last_conv(model.backbone)
            if target_conv is None:
                print("No convolutional layer found...")
                return

            print(f"Target layer for GradCAM: {target_conv}")

            wrapped_model = _SingleTensorWrapper(model.backbone).to(model.device)
            wrapped_model.eval()

            cam = GradCAM(model=wrapped_model, target_layers=[target_conv])

            fig, axes = plt.subplots(
                2, self.config.model.num_classes, figsize=(24, 8)
            )

            for cls_idx in range(self.config.model.num_classes):
                if cls_idx not in sample_images:
                    axes[0, cls_idx].axis('off')
                    axes[1, cls_idx].axis('off')
                    continue

                sample = sample_images[cls_idx]
                img = sample["image"].clone().detach().unsqueeze(0).to(
                    model.device, dtype=torch.float32
                )
                img.requires_grad_(True)

                true_label = int(sample["label"])
                pred_label = int(sample["pred"])
                targets = [ClassifierOutputTarget(pred_label)]

                with torch.enable_grad():
                    grayscale_cam = cam(input_tensor=img, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]

                img_np = sample["image"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)

                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                axes[0, cls_idx].imshow(img_np)
                axes[0, cls_idx].set_title(
                    f"{self.config.model.label_classes[cls_idx]}\nTrue: {self.config.model.label_classes[true_label]}",
                    fontsize=10
                )
                axes[0, cls_idx].axis('off')

                axes[1, cls_idx].imshow(visualization)
                axes[1, cls_idx].set_title(
                    f"Pred: {self.config.model.label_classes[pred_label]}",
                    fontsize=10,
                    color='green' if true_label == pred_label else 'red'
                )
                axes[1, cls_idx].axis('off')

            plt.suptitle("GradCAM Visualizations", fontsize=16, y=1.02)
            plt.tight_layout()

            print("GradCAM visualizations completed successfully!")

        except Exception as e:
            print(f"Warning: GradCAM visualization failed: {str(e)}")
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass
            try:
                del cam
            except Exception:
                pass

    def _find_last_conv(self, module: nn.Module):
        
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv
