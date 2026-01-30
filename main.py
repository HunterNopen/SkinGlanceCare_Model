import os
import argparse
import torch
import pytorch_lightning as pl

from config import Config
from data import ISICDataModule
from models import SkinLesionClassifier
from utils import TrainerFactory

def setup_reproducibility():
    pl.seed_everything(5, workers=True, verbose=False)
    torch.set_float32_matmul_precision("medium")


def print_header():
    print("\n" + "=" * 70)
    print(" " * 15 + "SKIN LESION CLASSIFICATION - FINAL")
    print("=" * 70)


def print_gpu_info(config: Config):
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")

        if torch.cuda.is_bf16_supported():
            print("Precision: BF16-mixed")
        else:
            config.training.precision = "16-mixed"
            print("Precision: FP16-mixed")
    else:
        print("WARNING: No GPU detected!")
        config.training.precision = "32"


def train_model(config: Config, datamodule: ISICDataModule, trainer: pl.Trainer, args):
   
    print("\n" + "=" * 70)
    print(" " * 20 + "STARTING TRAINING")
    print("=" * 70 + "\n")

    model = SkinLesionClassifier(config, class_counts=datamodule.class_counts)

    ckpt_path = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_path = args.checkpoint
        print(f"Resuming from checkpoint: {ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    best_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_checkpoint = callback.best_model_path
            best_score = callback.best_model_score
            print(f"\nBest checkpoint: {best_checkpoint}")
            print(f"Best val_composite score: {best_score:.4f}")
            break

    return best_checkpoint


def test_model(config: Config, datamodule: ISICDataModule, trainer: pl.Trainer, checkpoint_path: str):
   
    print("\n" + "=" * 70)
    print(" " * 20 + "STARTING TESTING")
    print("=" * 70 + "\n")

    if checkpoint_path is None:
        print("ERROR: --checkpoint required for testing")
        return

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    model = SkinLesionClassifier.load_from_checkpoint(
        checkpoint_path,
        config=config,
        class_counts=datamodule.class_counts
    )

    trainer.test(model, datamodule=datamodule)


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--train", "-t", action="store_true", default=False)
    parser.add_argument("-s", "--skip_train_pipeline", action="store_false", dest="train")
    parser.add_argument("--test", action="store_true", default=True)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)

    args = parser.parse_args()

    if not args.train and not args.test:
        parser.error("At least one of --train or --test must be specified")

    setup_reproducibility()
    print_header()

    config = Config()
    config.inference.cancer_threshold = args.threshold

    config.display()
    print_gpu_info(config)

    print("\nSetting up data module...")
    datamodule = ISICDataModule(config)
    datamodule.setup()

    print("Creating trainer...")
    trainer = TrainerFactory.create_trainer(config)

    best_checkpoint = None

    if args.train:
        best_checkpoint = train_model(config, datamodule, trainer, args)
        if best_checkpoint:
            args.checkpoint = best_checkpoint

    if args.test:
        checkpoint_path = args.checkpoint or best_checkpoint
        test_model(config, datamodule, trainer, checkpoint_path)

    print("\n" + "=" * 70)
    print(" " * 15 + "PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
