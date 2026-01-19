import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class MaxRecallLoss(nn.Module):

    def __init__(self, class_counts: np.ndarray, config: Config):
        super().__init__()

        self.cfg = config
        self.cancer_classes = set(config.model.cancer_classes)
        self.num_classes = config.model.num_classes

        self.register_buffer("base_weight", self._compute_class_weights(class_counts))
        self._print_configuration()

    def _compute_class_weights(self, class_counts: np.ndarray) -> torch.Tensor:
        
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / torch.sqrt(counts + 1)
        weights = weights / weights.sum() * len(weights)

        return weights

    def _print_configuration(self):

        print(f"\n{'=' * 60}")
        print("Recall-Maximizing Loss Configuration:")
        print(f"  FN multiplier (cancer): {self.cfg.loss.fn_multiplier}")
        print(f"  FN multiplier (MEL): {self.cfg.loss.mel_fn_multiplier}")
        print(f"  Recall loss weight: {self.cfg.loss.recall_loss_weight}")
        print(f"  Training temperature: {self.cfg.loss.training_temperature}")
        print(f"  Hard example mining: {self.cfg.loss.hard_example_mining}")
        print(f"{'=' * 60}\n")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        device = logits.device
        scaled_logits = logits / self.cfg.loss.training_temperature

        smoothed_targets = self._create_asymmetric_labels(targets, device)

        log_probs = F.log_softmax(scaled_logits, dim=1)
        ce_loss = -(smoothed_targets * log_probs).sum(dim=1)

        sample_weights = self.base_weight.to(device)[targets]
        ce_loss = ce_loss * sample_weights

        preds = torch.argmax(logits, dim=1)
        fn_multipliers = self._compute_fn_multipliers(targets, preds, device)
        ce_loss = ce_loss * fn_multipliers

        if self.cfg.loss.hard_example_mining:
            hard_weights = self._compute_hard_example_weights(ce_loss, targets, preds)
            ce_loss = ce_loss * hard_weights

        base_loss = ce_loss.mean()
        recall_loss = self._compute_recall_loss(logits, targets, device)

        total_loss = base_loss + self.cfg.loss.recall_loss_weight * recall_loss

        return total_loss

    def _create_asymmetric_labels(self, targets: torch.Tensor, device: torch.device) -> torch.Tensor:

        batch_size = targets.size(0)
        smoothed = torch.zeros(batch_size, self.num_classes, device=device)

        for i, t in enumerate(targets):
            t_val = t.item()

            if t_val in self.cancer_classes:
                smooth = self.cfg.loss.cancer_label_smoothing
            else:
                smooth = self.cfg.loss.benign_label_smoothing

            smoothed[i] = smooth / self.num_classes
            smoothed[i, t_val] = 1.0 - smooth + smooth / self.num_classes

            if t_val not in self.cancer_classes and smooth > 0:
                cancer_idx = list(self.cancer_classes)
                extra_cancer_mass = smooth * 0.5
                smoothed[i, cancer_idx] += extra_cancer_mass / len(cancer_idx)
                smoothed[i] = smoothed[i] / smoothed[i].sum()

        return smoothed

    def _compute_fn_multipliers(self, targets: torch.Tensor, preds: torch.Tensor, device: torch.device) -> torch.Tensor:
        
        multipliers = torch.ones(targets.size(0), device=device)

        is_cancer_true = torch.tensor(
            [t.item() in self.cancer_classes for t in targets], device=device
        )
        is_cancer_pred = torch.tensor(
            [p.item() in self.cancer_classes for p in preds], device=device
        )

        is_fn = is_cancer_true & ~is_cancer_pred
        multipliers[is_fn] = self.cfg.loss.fn_multiplier

        is_mel_fn = (targets == 0) & ~is_cancer_pred
        multipliers[is_mel_fn] = self.cfg.loss.mel_fn_multiplier

        return multipliers

    def _compute_hard_example_weights(self, losses: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:

        weights = torch.ones_like(losses)

        is_cancer = torch.tensor(
            [t.item() in self.cancer_classes for t in targets], device=losses.device
        )
        is_wrong = preds != targets

        hard_mask = is_cancer & is_wrong
        weights[hard_mask] = self.cfg.loss.hard_example_weight

        return weights

    def _compute_recall_loss(self, logits: torch.Tensor, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
        
        probs = F.softmax(logits, dim=1)

        cancer_idx = list(self.cancer_classes)
        cancer_probs = probs[:, cancer_idx].sum(dim=1)

        is_cancer = torch.tensor(
            [t.item() in self.cancer_classes for t in targets],
            dtype=torch.float32,
            device=device
        )

        soft_tp = (cancer_probs * is_cancer).sum()
        soft_fn = ((1 - cancer_probs) * is_cancer).sum()

        soft_recall = soft_tp / (soft_tp + soft_fn + 1e-8)
        recall_loss = 1.0 - soft_recall

        return recall_loss
