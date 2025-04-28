import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

class MRNet3(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model1 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model2 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model3 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Add batch normalization for each view's feature maps
        self.bn_view1 = nn.BatchNorm2d(256)  # AlexNet features output 256 channels
        self.bn_view2 = nn.BatchNorm2d(256)
        self.bn_view3 = nn.BatchNorm2d(256)
        
        # Add dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.3) 
        self.dropout_view2 = nn.Dropout(p=0.3)
        self.dropout_view3 = nn.Dropout(p=0.3)

        self.classifier1 = nn.Linear(int(256*3), 256)
        self.bn1 = nn.BatchNorm1d(256)  # BN after classifier1
        self.dropout = nn.Dropout(p=0.3) # test
        self.activation = nn.ReLU() 
        self.classifier2 = nn.Linear(256, 1)

    def forward(self, x, original_slices):
        
        view_features = []
        
        for view in range(3):
            
            x_view = x[view]  # [B, S_max, 3, 224, 224]
            B, S_max, _, H, W = x_view.shape
            x_view = x_view.view(B * S_max, 3, H, W)
            
            if view == 0:
                features = self.model1.features(x_view)
                features = self.bn_view1(features)
            elif view == 1:
                features = self.model2.features(x_view)
                features = self.bn_view2(features)
            else:
                features = self.model3.features(x_view)
                features = self.bn_view3(features)
            
            features = self.gap(features).view(B, S_max, 256)  # [B, S_max, 256]
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_features = torch.max(features, dim=1)[0]  # [B, 256]
            
            if view == 0:
                max_features = self.dropout_view1(max_features)
            elif view == 1:
                max_features = self.dropout_view2(max_features)
            else:
                max_features = self.dropout_view3(max_features)
            
            view_features.append(max_features)
        
        # Concatenate features from all views
        x_stacked = torch.cat(view_features, dim=1)  # [B, 1536]
        
        # Fully connected layers with BN
        x_stacked = self.classifier1(x_stacked)  # [B, 256]
        x_stacked = self.bn1(x_stacked)  # Apply batch normalization
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)  # [B, 1]
        
        return x_stacked