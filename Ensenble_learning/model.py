import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights, ResNet18_Weights

class MRNetAlex(nn.Module):
    """Model 1: AlexNet-based model with separate classifiers per view."""
    def __init__(self, use_batchnorm=False):
        super().__init__()
        self.model1 = models.alexnet(weights=AlexNet_Weights.DEFAULT)  # Axial
        self.model2 = models.alexnet(weights=AlexNet_Weights.DEFAULT)  # Coronal
        self.model3 = models.alexnet(weights=AlexNet_Weights.DEFAULT)  # Sagittal
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.use_batchnorm = use_batchnorm
        n = 0.15
        self.dropout_view1 = nn.Dropout(p=n)
        self.dropout_view2 = nn.Dropout(p=n)
        self.dropout_view3 = nn.Dropout(p=n)

        # Classifiers for each view
        classifier_layers_axial = [nn.Linear(256, 256)]
        if self.use_batchnorm:
            classifier_layers_axial.append(nn.BatchNorm1d(256))
        self.classifier1_axial = nn.Sequential(*classifier_layers_axial)
        self.classifier1_coronal = nn.Sequential(*[nn.Linear(256, 256)] + ([nn.BatchNorm1d(256)] if self.use_batchnorm else []))
        self.classifier1_sagittal = nn.Sequential(*[nn.Linear(256, 256)] + ([nn.BatchNorm1d(256)] if self.use_batchnorm else []))
        self.classifier2_axial = nn.Linear(256, 1)
        self.classifier2_coronal = nn.Linear(256, 1)
        self.classifier2_sagittal = nn.Linear(256, 1)

    def forward(self, x):
        # Not implemented as it's not needed for the ensemble
        pass

class MRNetResNet(nn.Module):
    """Model 2: ResNet18-based model with feature concatenation."""
    def __init__(self):
        super().__init__()
        self.model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model1 = nn.Sequential(*list(self.model1.children())[:-1])  # Axial
        self.model2 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model2 = nn.Sequential(*list(self.model2.children())[:-1])  # Coronal
        self.model3 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model3 = nn.Sequential(*list(self.model3.children())[:-1])  # Sagittal
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout_view1 = nn.Dropout(p=0.7)
        self.dropout_view2 = nn.Dropout(p=0.7)
        self.dropout_view3 = nn.Dropout(p=0.7)
        self.classifier1 = nn.Linear(512 * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)
        self.activation = nn.ReLU()
        self.classifier2 = nn.Linear(256, 1)

    def forward(self, x, original_slices):
        # Not implemented as it's not needed for the ensemble
        pass

class EnsembleMRNet(nn.Module):
    """Ensemble model combining CNNs from MRNetAlex and MRNetResNet with a new dense classifier."""
    def __init__(self, model1_path, model2_path, device):
        super().__init__()
        # Initialize base models
        self.model_alex = MRNetAlex()
        self.model_resnet = MRNetResNet()
        
        # Load pre-trained weights
        self.model_alex.load_state_dict(torch.load(model1_path, map_location=device, weights_only=True))
        self.model_resnet.load_state_dict(torch.load(model2_path, map_location=device, weights_only=True))
        
        # Freeze CNN parts
        for model in [self.model_alex.model1, self.model_alex.model2, self.model_alex.model3]:
            for param in model.features.parameters():
                param.requires_grad = False
        for model in [self.model_resnet.model1, self.model_resnet.model2, self.model_resnet.model3]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Define pooling layers consistent with original models
        self.gap_max = nn.AdaptiveMaxPool2d(1)  # For AlexNet
        self.gap_avg = nn.AdaptiveAvgPool2d(1)  # For ResNet18
        
        # Define dropout layers for each backbone per view
        self.dropout_alex_view1 = nn.Dropout(p=0.15)
        self.dropout_alex_view2 = nn.Dropout(p=0.15)
        self.dropout_alex_view3 = nn.Dropout(p=0.15)
        self.dropout_resnet_view1 = nn.Dropout(p=0.7)
        self.dropout_resnet_view2 = nn.Dropout(p=0.7)
        self.dropout_resnet_view3 = nn.Dropout(p=0.7)
        
        # New dense classifier
        self.dense = nn.Sequential(
            nn.Linear(2304, 1024),  # Input: 2304, Output: 512           
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            ###wasn't there in the best model 
            nn.Linear(1024, 512),   # Input: 1024, Output: 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            ###wasn't there in the best model
            nn.Linear(512, 1)      # Output: 1 for binary classification
        )

    def forward(self, padded_views, original_slices):
        """Forward pass: Extract features from both CNNs, concatenate, and classify."""
        B = padded_views[0].shape[0]
        view_features = []
        
        for view in range(3):
            x_view = padded_views[view]  # [B, S_max, 3, 224, 224]
            S_max = x_view.shape[1]
            x_view_flat = x_view.view(B * S_max, 3, 224, 224)
            
            # AlexNet features
            if view == 0:
                feat_alex = self.model_alex.model1.features(x_view_flat)  # [B*S_max, 256, 6, 6]
            elif view == 1:
                feat_alex = self.model_alex.model2.features(x_view_flat)
            else:
                feat_alex = self.model_alex.model3.features(x_view_flat)
            feat_alex = self.gap_max(feat_alex).view(B, S_max, 256)
            mask = torch.arange(S_max, device=feat_alex.device).expand(B, S_max) < original_slices[view].unsqueeze(1)
            feat_alex = feat_alex.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_feat_alex = torch.max(feat_alex, dim=1)[0]  # [B, 256]
            
            # ResNet18 features
            if view == 0:
                feat_resnet = self.model_resnet.model1(x_view_flat)  # [B*S_max, 512, 7, 7]
            elif view == 1:
                feat_resnet = self.model_resnet.model2(x_view_flat)
            else:
                feat_resnet = self.model_resnet.model3(x_view_flat)
            feat_resnet = self.gap_avg(feat_resnet).view(B, S_max, 512)
            feat_resnet = feat_resnet.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_feat_resnet = torch.max(feat_resnet, dim=1)[0]  # [B, 512]
            
            # Apply dropout
            if view == 0:
                max_feat_alex = self.dropout_alex_view1(max_feat_alex)
                max_feat_resnet = self.dropout_resnet_view1(max_feat_resnet)
            elif view == 1:
                max_feat_alex = self.dropout_alex_view2(max_feat_alex)
                max_feat_resnet = self.dropout_resnet_view2(max_feat_resnet)
            else:
                max_feat_alex = self.dropout_alex_view3(max_feat_alex)
                max_feat_resnet = self.dropout_resnet_view3(max_feat_resnet)
            
            # Concatenate features for this view
            combined_feat = torch.cat([max_feat_alex, max_feat_resnet], dim=1)  # [B, 768]
            view_features.append(combined_feat)
        
        # Concatenate all views
        all_features = torch.cat(view_features, dim=1)  # [B, 2304]
        
        # Dense classifier
        logits = self.dense(all_features)  # [B, 1]
        return logits