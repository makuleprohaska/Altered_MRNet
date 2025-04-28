import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class MRNet3(nn.Module):
    def __init__(self):
        super().__init__()
            
        # Load pretrained EfficientNet-B0 from torchvision
        self.model1 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model2 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model3 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the classifier head to get feature extractor
        self.model1.classifier = nn.Identity()  # EfficientNet-B0 outputs 1280 features
        self.model2.classifier = nn.Identity()
        self.model3.classifier = nn.Identity()
        
        # Enhanced fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

    def forward(self, x, original_slices):
        view_features = []
        
        for view in range(3):
            x_view = x[view]  # [B, S_max, 3, 224, 224]
            B, S_max, _, H, W = x_view.shape
            x_view = x_view.view(B * S_max, 3, H, W)
            
            if view == 0:
                features = self.model1(x_view)  # [B * S_max, 1280]
            elif view == 1:
                features = self.model2(x_view)
            else:
                features = self.model3(x_view)
            
            features = features.view(B, S_max, 1280)  # [B, S_max, 1280]
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_features = torch.max(features, dim=1)[0]  # [B, 1280]
            
            view_features.append(max_features)
        
        # Concatenate features from all views
        x_stacked = torch.cat(view_features, dim=1)  # [B, 1280 * 3 = 3840]
        
        # Pass through the enhanced classifier
        output = self.classifier(x_stacked)  # [B, 1]
        
        return output