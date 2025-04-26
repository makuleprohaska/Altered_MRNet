import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

class MRNet3(nn.Module):
    
    def __init__(self,use_batchnorm=True):
        super().__init__()
        self.model1 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model2 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model3 = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.gap = nn.AdaptiveMaxPool2d(1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.use_batchnorm = use_batchnorm
        
        # Dropout for each view's features
        self.dropout_view1 = nn.Dropout(p=0.25)
        self.dropout_view2 = nn.Dropout(p=0.25)
        self.dropout_view3 = nn.Dropout(p=0.25)
        print("Dropout 0.25")


        classifier_layers_axial = [nn.Linear(256, 256)]
        if self.use_batchnorm:
            classifier_layers_axial.append(nn.BatchNorm1d(256))
        self.classifier1_axial = nn.Sequential(*classifier_layers_axial)

        classifier_layers_coronal = [nn.Linear(256, 256)]
        if self.use_batchnorm:
            classifier_layers_coronal.append(nn.BatchNorm1d(256))
        self.classifier1_coronal = nn.Sequential(*classifier_layers_coronal)

        classifier_layers_sagittal = [nn.Linear(256, 256)]
        if self.use_batchnorm:
            classifier_layers_sagittal.append(nn.BatchNorm1d(256))
        self.classifier1_sagittal = nn.Sequential(*classifier_layers_sagittal)


        # Separate classifier2 for each view
        self.classifier2_axial = nn.Linear(256, 1)
        self.classifier2_coronal = nn.Linear(256, 1)
        self.classifier2_sagittal = nn.Linear(256, 1)


    #New forward pass to deal with batch normalisation

    def forward(self, x):
    
        # Separate by view
        axial_views    = [sample[0] for sample in x]
        coronal_views  = [sample[1] for sample in x]
        sagittal_views = [sample[2] for sample in x]

        def process_view(view_list, model, dropout, classifier1, classifier2):
            features = []
            for view in view_list:
                slices, c, h, w = view.size()  # [num_slices, 3, 224, 224]
                view = view.view(slices, c, h, w).to(next(model.parameters()).device)
                feat = model.features(view)                     # [slices, 256, 6, 6]
                feat = self.gap(feat).view(slices, 256)         # [slices, 256]
                feat = torch.max(feat, dim=0)[0]                # [256]
                feat = dropout(feat)
                features.append(feat)
            features = torch.stack(features)                    # [batch_size, 256]
            features = classifier1(features)                    # [batch_size, 256]
            logits = classifier2(features)                      # [batch_size, 1]
            return logits

        logit_axial    = process_view(axial_views,    self.model1, self.dropout_view1, self.classifier1_axial, self.classifier2_axial)
        logit_coronal  = process_view(coronal_views,  self.model2, self.dropout_view2, self.classifier1_coronal, self.classifier2_coronal)
        logit_sagittal = process_view(sagittal_views, self.model3, self.dropout_view3, self.classifier1_sagittal, self.classifier2_sagittal)

        logits = torch.stack([logit_axial, logit_coronal, logit_sagittal], dim=0)  # [3, batch_size, 1]
        probs = torch.sigmoid(logits)                                              # [3, batch_size, 1]
        majority_prob = torch.mean(probs, dim=0)                                   # [batch_size, 1]
    
        return majority_prob