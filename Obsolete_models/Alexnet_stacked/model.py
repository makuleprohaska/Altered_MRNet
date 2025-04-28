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
        
        # Add dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.15) # 0.1 -> 0.2 -> 0.1
        self.dropout_view2 = nn.Dropout(p=0.15) #
        self.dropout_view3 = nn.Dropout(p=0.15) #

        #Network head 
        self.classifier1 = nn.Linear(int(256*3), 256)
        self.dropout = nn.Dropout(p=0.1) # test
        self.activation = nn.ReLU() 
        self.classifier2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: list of [axial, coronal, sagittal] for each sample in batch
        batch_size = len(x)  # Number of samples (e.g., 4)
        print(f"batch_size: {batch_size}")
        batch_features = []
        
        for sample_views in x:  # Process each sample
            x_1, x_2, x_3 = sample_views[0], sample_views[1], sample_views[2]  # [slices, 3, 224, 224]
            
            slices, c, h, w = x_1.size()  # slices can vary per sample
            x_1 = x_1.view(slices, c, h, w)  # [slices, 3, 224, 224]
            x_1 = self.model1.features(x_1)
            x_1 = self.gap(x_1).view(slices, 256)  # [slices, 256] // average pooling [N, 256, 6, 6]-> [N, 256, 1, 1] .view -> [N, 256]
            x_1 = torch.max(x_1, 0)[0]  # [256]
            x_1 = self.dropout_view1(x_1) # Regularisation

            slices, c, h, w = x_2.size()
            x_2 = x_2.view(slices, c, h, w)
            x_2 = self.model2.features(x_2)
            x_2 = self.gap(x_2).view(slices, 256)
            x_2 = torch.max(x_2, 0)[0]
            x_2 = self.dropout_view2(x_2)

            slices, c, h, w = x_3.size()
            x_3 = x_3.view(slices, c, h, w)
            x_3 = self.model3.features(x_3)
            x_3 = self.gap(x_3).view(slices, 256)
            x_3 = torch.max(x_3, 0)[0]
            x_3 = self.dropout_view3(x_3)
            
            x_stacked = torch.cat((x_1, x_2, x_3), dim=0)  # [768]
            batch_features.append(x_stacked)
        
        x_stacked = torch.stack(batch_features)  # [batch_size, 768]
        x_stacked = self.classifier1(x_stacked)
        x_stacked = self.dropout(x_stacked)
        x_stacked = self.activation(x_stacked)
        x_stacked = self.classifier2(x_stacked)  # [batch_size, 1]
        
        return x_stacked
    


"""CHANGES FROM ALEXNET 

1) Dropout 0.1 -> 0.2 -> 0.1
2) Batch size 4 -> 16 (in loader.py)
3) Learning rate and weight decay tuned by hand when running model

"""