import torch
from typing import Tuple


class MetricsCalculator:
    
    @staticmethod
    def melanoma_vs_rest_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:

        binary_labels = (labels == 0).long()
        binary_preds = (preds == 0).long()

        tp = ((binary_preds == 1) & (binary_labels == 1)).sum().float()
        fp = ((binary_preds == 1) & (binary_labels == 0)).sum().float()
        fn = ((binary_preds == 0) & (binary_labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1.item(), precision.item(), recall.item()

    @staticmethod
    def cancer_vs_benign_metrics(preds: torch.Tensor, labels: torch.Tensor, cancer_classes: tuple) -> Tuple[float, float, float]:

        cancer_set = set(cancer_classes)

        binary_labels = torch.tensor(
            [1 if l.item() in cancer_set else 0 for l in labels]
        )
        binary_preds = torch.tensor(
            [1 if p.item() in cancer_set else 0 for p in preds]
        )

        tp = ((binary_preds == 1) & (binary_labels == 1)).sum().float()
        fp = ((binary_preds == 1) & (binary_labels == 0)).sum().float()
        fn = ((binary_preds == 0) & (binary_labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1.item(), precision.item(), recall.item()

    @staticmethod
    def analyze_cancer_thresholds(probs: torch.Tensor, labels: torch.Tensor, cancer_classes: tuple):
        
        cancer_idx = list(cancer_classes)
        cancer_probs = probs[:, cancer_idx].sum(dim=1)
        is_cancer = torch.tensor([l.item() in cancer_idx for l in labels])

        print(f"{'Thresh':<8} {'Cancer Rec':<12} {'Cancer Prec':<12} {'Cancer F1':<10}")
        print("-" * 42)

        for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
            pred_cancer = cancer_probs >= threshold
            tp = (pred_cancer & is_cancer).sum().float()
            fp = (pred_cancer & ~is_cancer).sum().float()
            fn = (~pred_cancer & is_cancer).sum().float()

            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            print(f"{threshold:<8.2f} {recall:<12.4f} {precision:<12.4f} {f1:<10.4f}")
