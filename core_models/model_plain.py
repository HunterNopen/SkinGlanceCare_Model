import torch
import torch.nn as nn
from torchvision import models

from clean_efficientnet import Config

class SkinGlanceCareClassifierPlainOV(nn.Module):
    def __init__(self, cfg: Config, pretrained_backbone: bool = False):
        super().__init__()
        self.cfg = cfg
        self.model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        in_feats = self.model.classifier[1].in_features

        classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, cfg.num_classes)
        )

        self.model.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
def load_from_lightning_checkpoint(ckpt_path: str, cfg: Config):
    print("Loading checkpoint goes brrrrr....")
    
    model = SkinGlanceCareClassifierPlainOV(cfg, pretrained_backbone=False)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Successfully Loaded Checkpoint - NICE!")
    except RuntimeError as e:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Failed! But still loaded... {e}")

    model.eval()
    return model