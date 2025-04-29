import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights, ResNet18_Weights

class MRNet3_AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model2 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model3 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn_view1 = nn.BatchNorm2d(256)
        self.bn_view2 = nn.BatchNorm2d(256)
        self.bn_view3 = nn.BatchNorm2d(256)
        self.classifier1 = nn.Linear(256 * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.activation = nn.ReLU()
        self.classifier2 = nn.Linear(256, 1)

    def forward(self, x, original_slices):
        view_features = []
        for view in range(3):
            x_view = x[view]
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
            features = self.gap(features).view(B, S_max, 256)
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_features = torch.max(features, dim=1)[0]
            view_features.append(max_features)
        x_stacked = torch.cat(view_features, dim=1)
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.bn1(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)
        return x_stacked

class MRNet3_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model2 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model3 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model1 = nn.Sequential(*list(self.model1.children())[:-1])
        self.model2 = nn.Sequential(*list(self.model2.children())[:-1])
        self.model3 = nn.Sequential(*list(self.model3.children())[:-1])
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
        view_features = []
        for view in range(3):
            x_view = x[view]
            B, S_max, _, H, W = x_view.shape
            x_view = x_view.view(B * S_max, 3, H, W)
            if view == 0:
                features = self.model1(x_view)
            elif view == 1:
                features = self.model2(x_view)
            else:
                features = self.model3(x_view)
            features = self.gap(features).view(B, S_max, 512)
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_features = torch.max(features, dim=1)[0]
            if view == 0:
                max_features = self.dropout_view1(max_features)
            elif view == 1:
                max_features = self.dropout_view2(max_features)
            else:
                max_features = self.dropout_view3(max_features)
            view_features.append(max_features)
        x_stacked = torch.cat(view_features, dim=1)
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.bn1(x_stacked)
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)
        return x_stacked

class EnsembleModel(nn.Module):
    def __init__(self, alexnet_model_path, resnet_model_path):
        super().__init__()

        # Load AlexNet-based model
        self.alexnet_model = MRNet3_AlexNet()
        state_dict = torch.load(alexnet_model_path, map_location='cpu')
        self.alexnet_model.load_state_dict(state_dict)
        
        # Freeze AlexNet CNN parts
        for param in self.alexnet_model.model1.parameters():
            param.requires_grad = False
        for param in self.alexnet_model.model2.parameters():
            param.requires_grad = False
        for param in self.alexnet_model.model3.parameters():
            param.requires_grad = False

        # Load ResNet18-based model
        self.resnet_model = MRNet3_ResNet()
        state_dict = torch.load(resnet_model_path, map_location='cpu')
        self.resnet_model.load_state_dict(state_dict)
        
        # Freeze ResNet18 CNN parts
        for param in self.resnet_model.model1.parameters():
            param.requires_grad = False
        for param in self.resnet_model.model2.parameters():
            param.requires_grad = False
        for param in self.resnet_model.model3.parameters():
            param.requires_grad = False

        # New classifier for combined features (768 + 1536 = 2304)
        self.classifier1 = nn.Linear(768 + 1536, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.ReLU()
        self.classifier2 = nn.Linear(256, 1)

    def forward(self, x, original_slices):
        # x: [vol_227_axial, vol_227_coronal, vol_227_sagittal, vol_224_axial, vol_224_coronal, vol_224_sagittal]

        # AlexNet features (227x227 inputs)
        alexnet_input = x[:3]
        view_features_alexnet = []
        for view in range(3):
            x_view = alexnet_input[view]
            B, S_max, _, H, W = x_view.shape
            x_view = x_view.view(B * S_max, 3, H, W)
            if view == 0:
                features = self.alexnet_model.model1.features(x_view)
                features = self.alexnet_model.bn_view1(features)
            elif view == 1:
                features = self.alexnet_model.model2.features(x_view)
                features = self.alexnet_model.bn_view2(features)
            else:
                features = self.alexnet_model.model3.features(x_view)
                features = self.alexnet_model.bn_view3(features)
            features = self.alexnet_model.gap(features).view(B, S_max, 256)
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_features = torch.max(features, dim=1)[0]
            view_features_alexnet.append(max_features)
        x_stacked_alexnet = torch.cat(view_features_alexnet, dim=1)  # [B, 768]

        # ResNet18 features (224x224 inputs)
        resnet_input = x[3:]
        view_features_resnet = []
        for view in range(3):
            x_view = resnet_input[view]
            B, S_max, _, H, W = x_view.shape
            x_view = x_view.view(B * S_max, 3, H, W)
            if view == 0:
                features = self.resnet_model.model1(x_view)
            elif view == 1:
                features = self.resnet_model.model2(x_view)
            else:
                features = self.resnet_model.model3(x_view)
            features = self.resnet_model.gap(features).view(B, S_max, 512)
            s_indices = torch.arange(S_max, device=features.device).unsqueeze(0).expand(B, S_max)
            mask = s_indices < original_slices[view].unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(2), -float('inf'))
            max_features = torch.max(features, dim=1)[0]
            view_features_resnet.append(max_features)
        x_stacked_resnet = torch.cat(view_features_resnet, dim=1)  # [B, 1536]

        # Combine features
        x_combined = torch.cat([x_stacked_alexnet, x_stacked_resnet], dim=1)  # [B, 2304]

        # New classifier
        x = self.classifier1(x_combined)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.classifier2(x)
        return x