import torch
import torch.nn as nn
import timm  # For EfficientNet models

class MRNet3(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Replace ResNet18 with EfficientNet-B2
        self.model1 = timm.create_model('efficientnet_b2', pretrained=True)
        self.model2 = timm.create_model('efficientnet_b2', pretrained=True)
        self.model3 = timm.create_model('efficientnet_b2', pretrained=True)
        
        # Remove the classifier head to get feature extractor
        self.model1.classifier = nn.Identity()  # EfficientNet-B2 outputs 1408 features
        self.model2.classifier = nn.Identity()
        self.model3.classifier = nn.Identity()

        # EfficientNet-B2 includes global average pooling, so explicit GAP may not be needed
        # Keeping it for consistency with original code, but it’s effectively a no-op
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Add dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.3)
        self.dropout_view2 = nn.Dropout(p=0.3)
        self.dropout_view3 = nn.Dropout(p=0.3)
        
        # EfficientNet-B2 feature output is 1408 per model
        self.classifier1 = nn.Linear(1408 * 3, 256)  # Update for 1408 features per view
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.ReLU()
        self.dropout_final = nn.Dropout(p=0.3)
        self.classifier2 = nn.Linear(256, 1)

    def forward(self, x):
        # x is a list of 3 tensors (for axial, coronal, and sagittal)
        x_1 = x[0]
        x_2 = x[1]
        x_3 = x[2]
        
        # Process each input slice using EfficientNet-B2 model (remove batch dimension)
        x_1 = torch.squeeze(x_1, dim=0)
        x_1 = self.model1(x_1)  # Shape: (slices, 1408)
        x_1 = self.gap(x_1.view(x_1.size(0), -1, 1, 1)).view(x_1.size(0), -1)  # Ensure flat
        x_1 = torch.max(x_1, 0, keepdim=True)[0]  # Max pooling over slices
        x_1 = self.dropout_view1(x_1)

        x_2 = torch.squeeze(x_2, dim=0)
        x_2 = self.model2(x_2)
        x_2 = self.gap(x_2.view(x_2.size(0), -1, 1, 1)).view(x_2.size(0), -1)
        x_2 = torch.max(x_2, 0, keepdim=True)[0]
        x_2 = self.dropout_view2(x_2)

        x_3 = torch.squeeze(x_3, dim=0)
        x_3 = self.model3(x_3)
        x_3 = self.gap(x_3.view(x_3.size(0), -1, 1, 1)).view(x_3.size(0), -1)
        x_3 = torch.max(x_3, 0, keepdim=True)[0]
        x_3 = self.dropout_view3(x_3)
        
        # Concatenate the features from all 3 models
        x_stacked = torch.cat((x_1, x_2, x_3), dim=1)  # Shape: (1, 1408 * 3)
        
        # Pass through the classifier
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.dropout_final(x_stacked)
        x_stacked = self.classifier2(x_stacked)
        
        return x_stacked