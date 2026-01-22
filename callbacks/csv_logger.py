import csv
import torch
from pytorch_lightning.callbacks import Callback

from config import Config


class CSVMetricsCallback(Callback):

    def __init__(self, csv_path: str, config: Config):
        super().__init__()
        self.csv_path = csv_path
        self.config = config
        self.metrics_history = []

        self._initialize_csv()

    def _initialize_csv(self):
        headers = [
            "epoch", "train_loss", "train_acc", "train_f1",
            "val_loss", "val_acc", "val_f1",
            "val_cancer_recall", "val_cancer_precision", "val_cancer_f1",
            "val_mel_recall", "val_mel_precision",
            "learning_rate"
        ]

        for cls in self.config.model.label_classes:
            headers.extend([f"val_{cls}_precision", f"val_{cls}_recall"])

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def on_validation_epoch_end(self, trainer, pl_module):
        
        metrics = trainer.callback_metrics

        row = {
            "epoch": trainer.current_epoch,
            "train_loss": metrics.get("train_loss_epoch", 0),
            "train_acc": metrics.get("train_acc", 0),
            "train_f1": metrics.get("train_f1", 0),
            "val_loss": metrics.get("val_loss", 0),
            "val_acc": metrics.get("val_acc", 0),
            "val_f1": metrics.get("val_f1", 0),
            "val_cancer_recall": metrics.get("val_cancer_recall", 0),
            "val_cancer_precision": metrics.get("val_cancer_precision", 0),
            "val_cancer_f1": metrics.get("val_cancer_f1", 0),
            "val_mel_recall": metrics.get("val_mel_recall", 0),
            "val_mel_precision": metrics.get("val_mel_precision", 0),
            "learning_rate": trainer.optimizers[0].param_groups[0]['lr'],
        }

        for _, cls in enumerate(self.config.model.label_classes):
            row[f"val_{cls}_precision"] = metrics.get(f"val_{cls}_precision", 0)
            row[f"val_{cls}_recall"] = metrics.get(f"val_{cls}_recall", 0)

        self.metrics_history.append(row)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow({
                k: float(v) if torch.is_tensor(v) else v
                for k, v in row.items()
            })
