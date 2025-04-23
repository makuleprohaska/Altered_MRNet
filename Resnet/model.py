import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class MRNet3(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Replace AlexNet with ResNet18
        self.model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model2 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model3 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the original classification layer (fully connected layer)
        self.model1 = nn.Sequential(*list(self.model1.children())[:-1])
        self.model2 = nn.Sequential(*list(self.model2.children())[:-1])
        self.model3 = nn.Sequential(*list(self.model3.children())[:-1])

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to reduce spatial dimensions
        
        # Add dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.7)
        self.dropout_view2 = nn.Dropout(p=0.7)
        self.dropout_view3 = nn.Dropout(p=0.7)
        
        # The ResNet feature output will be of size 512 per model
        self.classifier1 = nn.Linear(512 * 3, 256)  # ResNet18 produces a 512-dimensional vector
        self.dropout = nn.Dropout(p=0.4)  # Existing dropout
        self.activation = nn.ReLU()
        self.classifier2 = nn.Linear(256, 1)

    # *** Modified: Added original_slices parameter and updated for batch processing ***
    def forward(self, x, original_slices):
        view_features = []
        for view in range(3):
            x_view = x[view]  # [B, S_max, 3, 256, 256]
            B, S_max, _, H, W = x_view.shape
            # Reshape for ResNet: treat all slices across batch as a single batch
            x_view = x_view.view(B * S_max, 3, H, W)
            if view == 0:
                features = self.model1(x_view)
            elif view == 1:
                features = self.model2(x_view)
            else:
                features = self.model3(x_view)
            # Apply GAP and reshape back to per-sample features
            features = self.gap(features).view(B, S_max, 512)  # [B, S_max, 512]
            # Mask padded slices with -inf to exclude them from max
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)  # [B, S_max]
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            # Max over slices per sample
            max_features = torch.max(features, dim=1)[0]  # [B, 512]
            # Apply dropout
            if view == 0:
                max_features = self.dropout_view1(max_features)
            elif view == 1:
                max_features = self.dropout_view2(max_features)
            else:
                max_features = self.dropout_view3(max_features)
            view_features.append(max_features)
        
        # Concatenate features from all views
        x_stacked = torch.cat(view_features, dim=1)  # [B, 1536]
        
        # Pass through the classifier
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)
        
        return x_stacked