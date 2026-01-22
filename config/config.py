from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    base_model: str = "efficientnet_b3"
    image_size: int = 300
    dropout_1: float = 0.3
    dropout_2: float = 0.2
    num_classes: int = 8

    label_classes: Tuple[str, ...] = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
    cancer_classes: Tuple[int, ...] = (0, 2, 3, 7)

    full_class_names: Dict[str, str] = None

    def __post_init__(self):
        if self.full_class_names is None:
            self.full_class_names = {
                "MEL": "Melanoma (malignant)",
                "NV": "Melanocytic Nevus (benign)",
                "BCC": "Basal Cell Carcinoma (malignant)",
                "AK": "Actinic Keratosis (pre-malignant)",
                "BKL": "Benign Keratosis",
                "DF": "Dermatofibroma (benign)",
                "VASC": "Vascular Lesion (benign)",
                "SCC": "Squamous Cell Carcinoma (malignant)",
            }


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""

    batch_size: int = 32
    max_epochs: int = 80
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    precision: str = "bf16-mixed"

    freeze_epochs: int = 10
    num_blocks_to_freeze: int = 3


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""

    path_train_images: str = "./SkinGlanceCare_Model/data/datasets/ISIC_2019_Training_Input"
    path_train_gt: str = "./SkinGlanceCare_Model/data/datasets/ISIC_2019_Training_GroundTruth.csv"
    path_train_meta: str = "./SkinGlanceCare_Model/data/datasets/ISIC_2019_Training_Metadata.csv"
    path_test_images: str = "./SkinGlanceCare_Model/data/datasets/ISIC_2019_Test_Input"
    path_test_gt: str = "./SkinGlanceCare_Model/data/datasets/ISIC_2019_Test_GroundTruth.csv"
    path_test_meta: str = "./SkinGlanceCare_Model/data/datasets/ISIC_2019_Test_Metadata.csv"
    output_dir: str = "./outputs"

    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    use_hair_removal: bool = False
    hair_removal_kernel: int = 17
    use_color_constancy: bool = True

    cancer_oversample_factor: float = 2.0


@dataclass
class LossConfig:
    """Loss function configuration for recall maximization"""

    fn_multiplier: float = 4.0
    mel_fn_multiplier: float = 6.0
    recall_loss_weight: float = 0.3
    cancer_label_smoothing: float = 0.01
    benign_label_smoothing: float = 0.15
    training_temperature: float = 1.5
    hard_example_mining: bool = True
    hard_example_weight: float = 2.0


@dataclass
class InferenceConfig:
    """Inference and evaluation configuration"""

    mc_dropout_samples: int = 10
    uncertainty_threshold: float = 0.3
    ood_prob_threshold: float = 0.4
    cancer_threshold: float = 0.50
    tta_enabled: bool = True
    tta_augments: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration"""

    log_every_n_steps: int = 25
    csv_log_path: str = "training_metrics.csv"


@dataclass
class Config:
    """Main configuration class combining all sub-configs"""

    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    loss: LossConfig = None
    inference: InferenceConfig = None
    logging: LoggingConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.loss is None:
            self.loss = LossConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

    def display(self):
        """Display configuration in formatted sections"""
        print("\n" + "=" * 70)
        print(" " * 20 + "CONFIGURATION")
        print("=" * 70)

        sections = {
            "Model": self.model.__dict__,
            "Training": self.training.__dict__,
            "Data": {k: v for k, v in self.data.__dict__.items() if k.startswith('path') or k.startswith('use')},
            "Loss": self.loss.__dict__,
            "Inference": self.inference.__dict__,
            "Logging": self.logging.__dict__,
        }

        for section_name, section_data in sections.items():
            print(f"\n[{section_name}]")
            for key, value in section_data.items():
                if not key.startswith('_') and key not in ['full_class_names']:
                    print(f"  {key}: {value}")

        print("\n" + "=" * 70 + "\n")
