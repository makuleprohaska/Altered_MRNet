
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

#might need to increase weight decay if still overfitting 

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

    def forward(self, x):
        # x is a list of 3 tensors (for axial, coronal, and sagittal)
        x_1 = x[0]
        x_2 = x[1]
        x_3 = x[2]
        
        # Process each input slice using ResNet model (first remove batch dimension)
        x_1 = torch.squeeze(x_1, dim=0)
        x_1 = self.model1(x_1)
        x_1 = self.gap(x_1).view(x_1.size(0), -1)  # Flatten after GAP
        x_1 = torch.max(x_1, 0, keepdim=True)[0]
        x_1 = self.dropout_view1(x_1)  # Apply dropout to view features

        x_2 = torch.squeeze(x_2, dim=0)
        x_2 = self.model2(x_2)
        x_2 = self.gap(x_2).view(x_2.size(0), -1)
        x_2 = torch.max(x_2, 0, keepdim=True)[0]
        x_2 = self.dropout_view2(x_2)  # Apply dropout to view features

        x_3 = torch.squeeze(x_3, dim=0)
        x_3 = self.model3(x_3)
        x_3 = self.gap(x_3).view(x_3.size(0), -1)
        x_3 = torch.max(x_3, 0, keepdim=True)[0]
        x_3 = self.dropout_view3(x_3)  # Apply dropout to view features
        
        # Concatenate the features from all 3 models
        x_stacked = torch.cat((x_1, x_2, x_3), dim=1)
        
        # Pass through the classifier
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)
        
        return x_stacked