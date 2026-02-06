import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models.classifier import SkinLesionClassifier
from data.padufes_datamodule import PadUfes20DataModule

pl.seed_everything(42, workers=True, verbose=False)
torch.set_float32_matmul_precision("medium")

class PadUfesKFoldTester:

    def __init__(self, checkpoint_path: str, config: Config, n_folds: int = 5):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.n_folds = n_folds

        print(f"\nLoading model from checkpoint: {checkpoint_path}")
        self.model = SkinLesionClassifier.load_from_checkpoint(
            checkpoint_path,
            config=config,
            class_counts=np.ones(config.model.num_classes),
        )
        self.model.eval()
        self.model.freeze()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.fold_results = []
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []

    def run_kfold_evaluation(self):

        print(f"\n{'=' * 70}")
        print(f"Starting {self.n_folds}-Fold Cross Validation on PAD-UFES-20")
        print(f"{'=' * 70}\n")

        for fold in range(self.n_folds):
            print(f"\n{'=' * 70}")
            print(f"Evaluating Fold {fold + 1}/{self.n_folds}")
            print(f"{'=' * 70}")

            datamodule = PadUfes20DataModule(
                config=self.config,
                fold=fold,
                n_splits=self.n_folds
            )
            datamodule.setup()

            fold_result = self._evaluate_fold(datamodule, fold)
            self.fold_results.append(fold_result)

        self._aggregate_and_report()

    def _evaluate_fold(self, datamodule: PadUfes20DataModule, fold: int) -> Dict:

        test_loader = datamodule.test_dataloader()

        fold_preds = []
        fold_labels = []
        fold_probs = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.config.inference.mc_dropout_samples > 1:
                    probs, _ = self.model.forward_with_mc_dropout(
                        images,
                        self.config.inference.mc_dropout_samples
                    )
                else:
                    logits = self.model(images)
                    probs = F.softmax(logits, dim=1)

                preds = torch.argmax(probs, dim=1)

                fold_preds.append(preds.cpu().numpy())
                fold_labels.append(labels.cpu().numpy())
                fold_probs.append(probs.cpu().numpy())

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(test_loader)}")

        fold_preds = np.concatenate(fold_preds)
        fold_labels = np.concatenate(fold_labels)
        fold_probs = np.concatenate(fold_probs)

        self.all_predictions.extend(fold_preds)
        self.all_labels.extend(fold_labels)
        self.all_probs.extend(fold_probs)

        fold_metrics = self._calculate_metrics(fold_preds, fold_labels, fold_probs, fold)

        return {
            "fold": fold,
            "predictions": fold_preds,
            "labels": fold_labels,
            "probabilities": fold_probs,
            "metrics": fold_metrics,
        }

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        fold: int) -> Dict:

        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
        recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)

        precision_per_class = precision_score(
            labels, predictions, average=None, zero_division=0, labels=range(self.config.model.num_classes)
        )
        recall_per_class = recall_score(
            labels, predictions, average=None, zero_division=0, labels=range(self.config.model.num_classes)
        )
        f1_per_class = f1_score(
            labels, predictions, average=None, zero_division=0, labels=range(self.config.model.num_classes)
        )

        # Cancer vs Benign metrics
        cancer_indices = set(self.config.model.cancer_classes)
        is_cancer_true = np.array([label in cancer_indices for label in labels])
        is_cancer_pred = np.array([pred in cancer_indices for pred in predictions])

        if is_cancer_true.sum() > 0:
            cancer_recall = recall_score(is_cancer_true, is_cancer_pred, zero_division=0)
            cancer_precision = precision_score(is_cancer_true, is_cancer_pred, zero_division=0)
            cancer_f1 = f1_score(is_cancer_true, is_cancer_pred, zero_division=0)
        else:
            cancer_recall = cancer_precision = cancer_f1 = 0.0

        is_mel_true = (labels == 0)
        is_mel_pred = (predictions == 0)

        if is_mel_true.sum() > 0:
            mel_recall = recall_score(is_mel_true, is_mel_pred, zero_division=0)
            mel_precision = precision_score(is_mel_true, is_mel_pred, zero_division=0)
            mel_f1 = f1_score(is_mel_true, is_mel_pred, zero_division=0)
        else:
            mel_recall = mel_precision = mel_f1 = 0.0

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "cancer_recall": cancer_recall,
            "cancer_precision": cancer_precision,
            "cancer_f1": cancer_f1,
            "melanoma_recall": mel_recall,
            "melanoma_precision": mel_precision,
            "melanoma_f1": mel_f1,
        }

        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  Cancer Recall: {cancer_recall:.4f}")
        print(f"  Cancer Precision: {cancer_precision:.4f}")
        print(f"  Melanoma Recall: {mel_recall:.4f}")

        return metrics

    def _aggregate_and_report(self):

        print(f"\n{'=' * 70}")
        print(f"K-FOLD CROSS VALIDATION RESULTS")
        print(f"{'=' * 70}\n")

        all_preds = np.array(self.all_predictions)
        all_labels = np.array(self.all_labels)
        all_probs = np.array(self.all_probs)

        overall_metrics = self._calculate_metrics(all_preds, all_labels, all_probs, fold=-1)

        fold_accuracies = [f["metrics"]["accuracy"] for f in self.fold_results]
        fold_f1s = [f["metrics"]["f1_macro"] for f in self.fold_results]
        fold_cancer_recalls = [f["metrics"]["cancer_recall"] for f in self.fold_results]
        fold_mel_recalls = [f["metrics"]["melanoma_recall"] for f in self.fold_results]

        print("\n" + "=" * 70)
        print("OVERALL RESULTS (Aggregated across all folds)")
        print("=" * 70)
        print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"Overall F1 (macro): {overall_metrics['f1_macro']:.4f}")
        print(f"Overall Cancer Recall: {overall_metrics['cancer_recall']:.4f}")
        print(f"Overall Cancer Precision: {overall_metrics['cancer_precision']:.4f}")
        print(f"Overall Melanoma Recall: {overall_metrics['melanoma_recall']:.4f}")

        print("\n" + "=" * 70)
        print("PER-FOLD STATISTICS")
        print("=" * 70)
        print(f"Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"F1 (macro): {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"Cancer Recall: {np.mean(fold_cancer_recalls):.4f} ± {np.std(fold_cancer_recalls):.4f}")
        print(f"Melanoma Recall: {np.mean(fold_mel_recalls):.4f} ± {np.std(fold_mel_recalls):.4f}")

        print("\n" + "=" * 70)
        print("PER-CLASS RESULTS (Overall)")
        print("=" * 70)
        for i, class_name in enumerate(self.config.model.label_classes):
            prec = overall_metrics["precision_per_class"][i]
            rec = overall_metrics["recall_per_class"][i]
            f1 = overall_metrics["f1_per_class"][i]
            is_cancer = "CANCER" if i in self.config.model.cancer_classes else "BENIGN"

            # Count samples
            n_samples = (all_labels == i).sum()

            print(f"\n{class_name} ({is_cancer}) - {n_samples} samples:")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1:        {f1:.4f}")

        self._generate_visualizations(all_preds, all_labels, overall_metrics)

        self._save_results(overall_metrics, fold_accuracies, fold_f1s, fold_cancer_recalls)

    def _generate_visualizations(self, predictions: np.ndarray, labels: np.ndarray,metrics: Dict):

        output_dir = Path(self.config.data.output_dir) / "padufes_kfold"
        output_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(labels, predictions, labels=range(self.config.model.num_classes))

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.config.model.label_classes,
            yticklabels=self.config.model.label_classes,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix - PAD-UFES-20 K-Fold CV")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix_kfold.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nConfusion matrix saved to: {output_dir / 'confusion_matrix_kfold.png'}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        class_names = list(self.config.model.label_classes)
        precision = metrics["precision_per_class"]
        recall = metrics["recall_per_class"]
        f1 = metrics["f1_per_class"]

        axes[0].bar(class_names, precision, color="steelblue")
        axes[0].set_title("Precision per Class")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(class_names, recall, color="coral")
        axes[1].set_title("Recall per Class")
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(class_names, f1, color="seagreen")
        axes[2].set_title("F1 Score per Class")
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "per_class_metrics_kfold.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Per-class metrics plot saved to: {output_dir / 'per_class_metrics_kfold.png'}")

    def _save_results(self, overall_metrics: Dict, fold_accuracies: List[float], fold_f1s: List[float], fold_cancer_recalls: List[float]):
        
        output_dir = Path(self.config.data.output_dir) / "padufes_kfold"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame({
            "Metric": [
                "Overall Accuracy",
                "Overall F1 (macro)",
                "Overall Cancer Recall",
                "Overall Cancer Precision",
                "Overall Melanoma Recall",
                "Mean Fold Accuracy",
                "Std Fold Accuracy",
                "Mean Fold F1",
                "Std Fold F1",
                "Mean Fold Cancer Recall",
                "Std Fold Cancer Recall",
            ],
            "Value": [
                overall_metrics["accuracy"],
                overall_metrics["f1_macro"],
                overall_metrics["cancer_recall"],
                overall_metrics["cancer_precision"],
                overall_metrics["melanoma_recall"],
                np.mean(fold_accuracies),
                np.std(fold_accuracies),
                np.mean(fold_f1s),
                np.std(fold_f1s),
                np.mean(fold_cancer_recalls),
                np.std(fold_cancer_recalls),
            ]
        })

        results_path = output_dir / f"kfold_results_{self.n_folds}folds.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        per_class_df = pd.DataFrame({
            "Class": self.config.model.label_classes,
            "Precision": overall_metrics["precision_per_class"],
            "Recall": overall_metrics["recall_per_class"],
            "F1": overall_metrics["f1_per_class"],
        })

        per_class_path = output_dir / f"per_class_results_{self.n_folds}folds.csv"
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Per-class results saved to: {per_class_path}")


def main():
    parser = argparse.ArgumentParser(description="K-Fold CV Testing on PAD-UFES-20")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/Python/projects/self/SkinGlanceCare/lightning_logs/skin_lesion_final/20260111_124444/checkpoints/epoch=13-val_f1=0.542-val_cancer_recall_thresh=0.968.ckpt",
        #required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=3,
        help="Number of folds for cross validation (default: 5)"
    )

    args = parser.parse_args()

    config = Config()

    tester = PadUfesKFoldTester(
        checkpoint_path=args.checkpoint,
        config=config,
        n_folds=args.n_folds
    )

    tester.run_kfold_evaluation()

    print("\nK-Fold Cross Validation completed!")

if __name__ == "__main__":
    main()
