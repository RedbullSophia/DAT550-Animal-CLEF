import torch
import torch.nn as nn
from timm import create_model

class ReIDNet(nn.Module):
    def __init__(self, backbone_name="mobilenetv3_small_100", embedding_dim=512, device="cuda"):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.to(device)  # ✅ move backbone first

        # Use dummy input on the correct device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = self.backbone(dummy_input)

        # Define embedding layer using output feature dim
        self.embedding = nn.Linear(out.shape[1], embedding_dim)
        self.embedding.to(device)  # ✅ move embedding too

        self.to(device)  # ✅ Just in case anything's still on CPU

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x
