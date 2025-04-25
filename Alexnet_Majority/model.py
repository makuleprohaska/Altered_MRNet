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
        self.gap = nn.AdaptiveMaxPool2d(1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.15)
        self.dropout_view2 = nn.Dropout(p=0.15) # changed dropout 
        self.dropout_view3 = nn.Dropout(p=0.15)

        # Separate classifier1 for each view
        self.classifier1_axial = nn.Sequential(
            nn.Linear(256, 256),
            # nn.GroupNorm(num_groups=32, num_channels=256),
        )
        self.classifier1_coronal = nn.Sequential(
            nn.Linear(256, 256),
            # nn.GroupNorm(num_groups=32, num_channels=256),
        )
        self.classifier1_sagittal = nn.Sequential(
            nn.Linear(256, 256),
            # nn.GroupNorm(num_groups=32, num_channels=256), #just for my Mac
        )

        # Separate classifier2 for each view
        self.classifier2_axial = nn.Linear(256, 1)
        self.classifier2_coronal = nn.Linear(256, 1)
        self.classifier2_sagittal = nn.Linear(256, 1)

    def forward(self, x):
        # x: list of [axial, coronal, sagittal] for each sample in batch
        batch_size = len(x)  # Number of samples
        batch_predictions = []
        
        for sample_views in x:  # Process each sample
            x_1, x_2, x_3 = sample_views[0], sample_views[1], sample_views[2]  # [slices, 3, 224, 224]
            
            # Axial view
            slices, c, h, w = x_1.size()
            x_1 = x_1.view(slices, c, h, w)  # [slices, 3, 224, 224]
            x_1 = self.model1.features(x_1)
            x_1 = self.gap(x_1).view(slices, 256)  # [slices, 256]
            x_1 = torch.max(x_1, 0)[0]  # [256]
            x_1 = self.dropout_view1(x_1)
            x_1 = x_1.unsqueeze(0)  # [1, 256] for GroupNorm
            x_1 = self.classifier1_axial(x_1)  # [1, 256]
            x_1 = x_1.squeeze(0)  # [256]
            logit_1 = self.classifier2_axial(x_1)  # [1]

            # Coronal view
            slices, c, h, w = x_2.size()
            x_2 = x_2.view(slices, c, h, w)
            x_2 = self.model2.features(x_2)
            x_2 = self.gap(x_2).view(slices, 256)
            x_2 = torch.max(x_2, 0)[0]
            x_2 = self.dropout_view2(x_2)
            x_2 = x_2.unsqueeze(0)  # [1, 256]
            x_2 = self.classifier1_coronal(x_2)  # [1, 256]
            x_2 = x_2.squeeze(0)  # [256]
            logit_2 = self.classifier2_coronal(x_2)  # [1]

            # Sagittal view
            slices, c, h, w = x_3.size()
            x_3 = x_3.view(slices, c, h, w)
            x_3 = self.model3.features(x_3)
            x_3 = self.gap(x_3).view(slices, 256)
            x_3 = torch.max(x_3, 0)[0]
            x_3 = self.dropout_view3(x_3)
            x_3 = x_3.unsqueeze(0)  # [1, 256]
            x_3 = self.classifier1_sagittal(x_3)  # [1, 256]
            x_3 = x_3.squeeze(0)  # [256]
            logit_3 = self.classifier2_sagittal(x_3)  # [1]

            # Majority voting with probabilities
            logits = torch.stack([logit_1, logit_2, logit_3], dim=0)  # [3, 1]
            probs = torch.sigmoid(logits)  # [3, 1]
            votes = (probs > 0.5).float()  # [3, 1]
            majority_prob = torch.mean(probs, dim=0)  # [1], average probability
            batch_predictions.append(majority_prob)

        # Stack predictions for the batch
        final_predictions = torch.stack(batch_predictions)  # [batch_size, 1]
        return final_predictions