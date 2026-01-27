import os
import torch
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config
from callbacks import CSVMetricsCallback


class TrainerFactory:

    @staticmethod
    def create_trainer(config: Config) -> pl.Trainer:

        TrainerFactory._setup_torch_backend()

        callbacks = TrainerFactory._create_callbacks(config)
        logger = TrainerFactory._create_logger()

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=config.training.precision,
            max_epochs=config.training.max_epochs,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=config.logging.log_every_n_steps,
            accumulate_grad_batches=config.training.accumulate_grad_batches,
            gradient_clip_val=config.training.gradient_clip_val,
            gradient_clip_algorithm="norm",
            deterministic=False,
            benchmark=True,
            enable_model_summary=True,
            inference_mode=False,
        )

        return trainer

    @staticmethod
    def _setup_torch_backend():
        
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _create_callbacks(config: Config) -> list:
        
        os.makedirs(config.data.output_dir, exist_ok=True)
        csv_path = os.path.join(config.data.output_dir, config.logging.csv_log_path)

        callbacks = [
            ModelCheckpoint(
                monitor="val_composite",
                mode="max",
                filename="{epoch:02d}-{val_f1:.3f}-{val_cancer_recall_thresh:.3f}",
                save_top_k=3,
                verbose=True,
            ),
            EarlyStopping(
                monitor="val_composite",
                patience=20,
                mode="max",
                min_delta=0.001,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            CSVMetricsCallback(csv_path=csv_path, config=config),
        ]

        return callbacks

    @staticmethod
    def _create_logger() -> TensorBoardLogger:
        
        return TensorBoardLogger(
            "lightning_logs",
            name="skin_lesion_final",
            version=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
