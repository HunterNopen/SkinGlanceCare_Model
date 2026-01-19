from typing import Tuple
import torch.nn as nn
from torchvision import models


class BackboneFactory:

    @staticmethod
    def create_backbone(model_name: str) -> Tuple[nn.Module, int]:
        
        if model_name == "efficientnet_b3":
            return BackboneFactory._create_efficientnet_b3()
        elif model_name == "efficientnet_b4":
            return BackboneFactory._create_efficientnet_b4()
        elif hasattr(models, model_name):
            return BackboneFactory._create_generic_model(model_name)
        else:
            print(f"Unknown model {model_name}, defaulting to efficientnet_b3")
            return BackboneFactory._create_efficientnet_b3()

    @staticmethod
    def _create_efficientnet_b3() -> Tuple[nn.Module, int]:

        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        print(f"Initialized EfficientNet-B3 backbone (features: {in_features})")
        return backbone, in_features

    @staticmethod
    def _create_efficientnet_b4() -> Tuple[nn.Module, int]:
        
        backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        print(f"Initialized EfficientNet-B4 backbone (features: {in_features})")
        return backbone, in_features

    @staticmethod
    def _create_generic_model(model_name: str) -> Tuple[nn.Module, int]:
        
        backbone = getattr(models, model_name)(weights="IMAGENET1K_V1")
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        print(f"Initialized {model_name} backbone (features: {in_features})")
        return backbone, in_features


class LayerFreezer:

    @staticmethod
    def freeze_early_blocks(backbone: nn.Module, num_blocks: int):
        
        if not hasattr(backbone, 'features'):
            return

        for idx, block in enumerate(backbone.features):
            if idx < num_blocks:
                for param in block.parameters():
                    param.requires_grad = False

    @staticmethod
    def unfreeze_all(backbone: nn.Module):
        for param in backbone.parameters():
            param.requires_grad = True
