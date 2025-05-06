# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class MRNet3(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        # three independent EfficientNet feature extractors
        self.e1 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.e2 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.e3 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # strip off their heads
        self.e1.classifier = nn.Identity()
        self.e2.classifier = nn.Identity()
        self.e3.classifier = nn.Identity()
        # dropout hyperparameter layer
        self.dropout = nn.Dropout(p=dropout)
        # new classifier on concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(1280*3, 512),
            nn.GroupNorm(num_groups=16, num_channels=512), ## !!! Switched to Group norm as had problem with feature size
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512,256),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256,1)
        )

    def update_dropout(self, new_p):
        self.dropout.p = new_p
        # also update internal classifier dropouts
        for m in self.classifier.modules():
            if isinstance(m, nn.Dropout):
                m.p = new_p

    def forward(self, vol_lists, original_slices):
        views_f = []
        for i, vol in enumerate(vol_lists):
            # vol: [B, S_max, 3, 224,224]
            B, S, C, H, W = vol.shape
            x = vol.view(B*S, C, H, W)
            if i==0: feats = self.e1(x)
            if i==1: feats = self.e2(x)
            if i==2: feats = self.e3(x)
            feats = feats.view(B, S, 1280)
            # mask invalid slices
            idx = torch.arange(S, device=feats.device).unsqueeze(0).expand(B,S)
            mask= idx < original_slices[i].unsqueeze(1)
            feats = feats.masked_fill(~mask.unsqueeze(2), -float('inf'))
            # max-pool over valid slices
            vfeat = torch.max(feats, dim=1)[0]  # [B,1280]
            views_f.append(vfeat)
        x = torch.cat(views_f, dim=1)  # [B,3840]
        x = self.dropout(x)
        out = self.classifier(x)
        return out
