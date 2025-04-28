import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class MRNet3(nn.Module):
    def __init__(self):
        super().__init__()
            
        # Load pretrained EfficientNet-B1 from torchvision
        self.model1 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model2 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model3 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the classifier head to get feature extractor
        self.model1.classifier = nn.Identity()  # EfficientNet-B1 outputs 1280 features
        self.model2.classifier = nn.Identity()
        self.model3.classifier = nn.Identity()
        
        # Dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.5)
        self.dropout_view2 = nn.Dropout(p=0.5)
        self.dropout_view3 = nn.Dropout(p=0.5)
        
        # Fully connected layers with batch normalization
        self.classifier1 = nn.Linear(1280 * 3, 1280)  # Concatenated features from 3 views (1280 * 3 = 3840)
        self.bn1 = nn.BatchNorm1d(1280)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.activation = nn.ReLU()
        
        self.classifier2 = nn.Linear(1280, 256)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.classifier3 = nn.Linear(256, 1)  # Fixed typo (Liner -> Linear)

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
            
            if view == 0:
                max_features = self.dropout_view1(max_features)
            elif view == 1:
                max_features = self.dropout_view2(max_features)
            else:
                max_features = self.dropout_view3(max_features)
            
            view_features.append(max_features)
        
        # Concatenate features from all views
        x_stacked = torch.cat(view_features, dim=1)  # [B, 1280 * 3 = 3840]
        
        # Fully connected layers with BN
        x_stacked = self.classifier1(x_stacked)  # [B, 1280]
        x_stacked = self.bn1(x_stacked)
        x_stacked = self.dropout1(x_stacked)
        x_stacked = self.activation(x_stacked)
        
        x_stacked = self.classifier2(x_stacked)  # [B, 256]
        x_stacked = self.dropout2(x_stacked)
        x_stacked = self.activation(x_stacked)
        
        x_stacked = self.classifier3(x_stacked)  # [B, 1]
        
        return x_stacked