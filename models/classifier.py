from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from config import Config
from .backbone import BackboneFactory, LayerFreezer
from losses import MaxRecallLoss
from utils.visualization import TestResultsVisualizer
from utils.metrics import MetricsCalculator

class ClassificationHead(nn.Module):

    def __init__(self, in_features: int, num_classes: int, dropout_1: float, dropout_2: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout_1),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SkinLesionClassifier(pl.LightningModule):

    def __init__(self, config: Config, class_counts: np.ndarray):
        super().__init__()
        self.cfg = config
        self.class_counts = class_counts
        self.save_hyperparameters(ignore=["config"])

        self.backbone, in_features = BackboneFactory.create_backbone(
            config.model.base_model
        )

        #self.head = ClassificationHead(in_features, config.model.num_classes, config.model.dropout_1, config.model.dropout_2)
        self.head = nn.Sequential(
            nn.Dropout(config.model.dropout_1),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(config.model.dropout_2),
            nn.Linear(512, config.model.num_classes),
        )

        self.loss_fn = MaxRecallLoss(class_counts=self.class_counts, config=self.cfg)

        self._setup_metrics()
        self._init_tracking_lists()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def forward_with_mc_dropout(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:

        ### BatchNormLayers affect std & mean of inference learnt during training. TRAIN => EVAL
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self(x)
                probs = F.softmax(logits, dim=1)
                preds.append(probs)

        self.eval()

        preds = torch.stack(preds, dim=0)
        mean_probs = preds.mean(dim=0)
        entropy = -(preds * torch.log(preds + 1e-8)).sum(dim=2)
        uncertainty = entropy.mean(dim=0)

        return mean_probs, uncertainty
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.learning_rate * 10,
            epochs=self.cfg.training.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.cfg.training.max_epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_epoch_start(self):
        
        if self.current_epoch < self.cfg.training.freeze_epochs:
            LayerFreezer.freeze_early_blocks(self.backbone, self.cfg.training.num_blocks_to_freeze)

        elif self.current_epoch == self.cfg.training.freeze_epochs:
            LayerFreezer.unfreeze_all(self.backbone)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        cancer_idx = list(self.cfg.model.cancer_classes)
        is_cancer_true = torch.tensor([t.item() in cancer_idx for t in y], device=y.device)
        is_cancer_pred = torch.tensor([p.item() in cancer_idx for p in preds], device=y.device)

        if is_cancer_true.sum() > 0:
            batch_cancer_recall = (is_cancer_true & is_cancer_pred).sum().float() / is_cancer_true.sum()
            self.log("train_cancer_recall", batch_cancer_recall, prog_bar=True, on_step=True, on_epoch=False)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_precision_per_class.update(preds, y)
        self.val_recall_per_class.update(preds, y)

        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        self.val_logits.append(logits.detach().cpu())

        self.log("val_loss", loss, prog_bar=False, on_epoch=True)

    def on_validation_epoch_end(self):

        val_f1 = self.val_f1.compute()
        val_acc = self.val_acc.compute()

        self.log("val_f1", val_f1, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        logits = torch.cat(self.val_logits)

        precision_per_class = self.val_precision_per_class.compute().cpu().numpy()
        recall_per_class = self.val_recall_per_class.compute().cpu().numpy()

        for i, cls in enumerate(self.cfg.model.label_classes):
            self.log(f"val_{cls}_precision", precision_per_class[i])
            self.log(f"val_{cls}_recall", recall_per_class[i])

        mel_f1, mel_prec, mel_rec = MetricsCalculator.melanoma_vs_rest_metrics(preds, labels)
        can_f1, can_prec, can_rec = MetricsCalculator.cancer_vs_benign_metrics(
            preds, labels, self.cfg.model.cancer_classes
        )

        self.log("val_mel_recall", mel_rec, prog_bar=True)
        self.log("val_mel_precision", mel_prec)
        self.log("val_cancer_recall", can_rec, prog_bar=True)
        self.log("val_cancer_precision", can_prec)
        self.log("val_cancer_f1", can_f1, prog_bar=True)

        all_probs = F.softmax(logits, dim=1)
        cancer_idx = list(self.cfg.model.cancer_classes)
        cancer_prob = all_probs[:, cancer_idx].sum(dim=1)
        is_cancer = torch.tensor([l.item() in cancer_idx for l in labels])

        thresh_pred_cancer = cancer_prob >= self.cfg.inference.cancer_threshold
        thresh_recall = (thresh_pred_cancer & is_cancer).sum().float() / (is_cancer.sum() + 1e-8)
        self.log("val_cancer_recall_thresh", thresh_recall, prog_bar=True)

        composite = 0.35 * val_f1 + 0.65 * thresh_recall
        self.log("val_composite", composite, prog_bar=True)

        if self.logger and hasattr(self.logger, 'experiment'):
            self._log_per_class_metrics(precision_per_class, recall_per_class)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision_per_class.reset()
        self.val_recall_per_class.reset()
        self.val_preds.clear()
        self.val_labels.clear()
        self.val_logits.clear()

    def test_step(self, batch, batch_idx):
        
        x, y = batch

        if self.cfg.inference.mc_dropout_samples > 1:
            probs, uncertainty = self.forward_with_mc_dropout(
                x, self.cfg.inference.mc_dropout_samples
            )
        else:
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            uncertainty = torch.zeros(x.size(0), device=x.device)

        preds = torch.argmax(probs, dim=1)

        if batch_idx % 20 == 0:
            for i, lbl in enumerate(y):
                cls = int(lbl.item())
                if cls not in self.sample_images:
                    self.sample_images[cls] = {
                        "image": x[i].detach().cpu().clone(),
                        "label": cls,
                        "pred": int(preds[i].item()),
                    }

        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        self.test_probs.append(probs.detach().cpu())
        self.test_uncertainties.append(uncertainty.detach().cpu())

        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_precision_per_class.update(preds, y)
        self.test_recall_per_class.update(preds, y)

    def on_test_epoch_end(self):

        visualizer = TestResultsVisualizer(self.cfg)

        acc = self.test_acc.compute()
        f1 = self.test_f1.compute()
        prec_per_class = self.test_precision_per_class.compute().cpu().numpy()
        rec_per_class = self.test_recall_per_class.compute().cpu().numpy()

        labels = torch.cat(self.test_labels)
        preds = torch.cat(self.test_preds)
        probs = torch.cat(self.test_probs)
        uncertainties = torch.cat(self.test_uncertainties)

        visualizer.print_test_summary(
            acc, f1, prec_per_class, rec_per_class, labels, preds, probs, uncertainties
        )

        visualizer.plot_confusion_matrix(
            preds.numpy(), labels.numpy(), self.cfg.data.output_dir, "test_standard"
        )

        print("Generating GradCAM visualizations...")
        with torch.set_grad_enabled(True):
            visualizer.generate_gradcam(self, self.sample_images)

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision_per_class.reset()
        self.test_recall_per_class.reset()
        self.test_preds.clear()
        self.test_labels.clear()
        self.test_probs.clear()
        self.test_uncertainties.clear()

    def _log_per_class_metrics(self, precision: np.ndarray, recall: np.ndarray):
        
        for i, cls_name in enumerate(self.cfg.model.label_classes):
            self.logger.experiment.add_scalars(
                f"PerClass/{cls_name}",
                {"precision": precision[i], "recall": recall[i]},
                self.current_epoch,
            )
    
    def _setup_metrics(self):
        
        nc = self.cfg.model.num_classes

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=nc, average="macro")

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=nc, average="macro")
        self.val_precision_per_class = torchmetrics.Precision(task="multiclass", num_classes=nc, average=None)
        self.val_recall_per_class = torchmetrics.Recall(task="multiclass", num_classes=nc, average=None)

        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=nc, average="macro")
        self.test_precision_per_class = torchmetrics.Precision(task="multiclass", num_classes=nc, average=None)
        self.test_recall_per_class = torchmetrics.Recall(task="multiclass", num_classes=nc, average=None)

    def _init_tracking_lists(self):
        
        self.val_preds: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []
        self.val_logits: List[torch.Tensor] = []

        self.test_preds: List[torch.Tensor] = []
        self.test_labels: List[torch.Tensor] = []
        self.test_probs: List[torch.Tensor] = []
        self.test_uncertainties: List[torch.Tensor] = []

        self.sample_images: Dict[int, Dict] = {}