import torch
import torch.nn as nn
from torchvision import models

# Define the MRNet class, a neural network for MRI volume classification
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the parent nn.Module class
        
        # Load a pretrained AlexNet model for feature extraction
        self.model = models.alexnet(pretrained=True)  # Pretrained on ImageNet, expects [3, 224, 224] inputs
        
        # Adaptive average pooling layer to reduce spatial dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)  # Output will be [256, 1, 1] per slice
        
        # Fully connected layer to produce a single output (binary classification)
        self.classifier = nn.Linear(256, 1)  # 256 input features (from AlexNet) to 1 output logit

    # Define the forward pass of the model
    def forward(self, x):
        # Input x has shape [1, s, 3, 224, 224] (batch_size=1, s slices, 3 channels, 224x224)
        x = torch.squeeze(x, dim=0)  # Remove batch dimension: [s, 3, 224, 224]
        # Note: This assumes batch_size=1; larger batches would break this
        
        # Pass each slice through AlexNet's feature extractor
        x = self.model.features(x)  # Output: [s, 256, h', w'] (in our case, [s, 256, 7, 7] after convolutions)
        
        # Apply global average pooling to each slice's feature map
        x = self.gap(x).view(x.size(0), -1)  # [s, 256, 1, 1] -> [s, 256] (flatten spatial dims)
        
        # Take the maximum feature value across all slices
        x = torch.max(x, 0, keepdim=True)[0]  # [s, 256] -> [1, 256] (max pooling over slices)
        
        # Pass the aggregated features through the classifier
        x = self.classifier(x)  # [1, 256] -> [1, 1] (single logit for the volume)
        
        return x  # Return the final prediction logit