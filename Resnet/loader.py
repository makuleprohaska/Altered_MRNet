import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms


INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

###This part was added to try with the MRNet3 model
class Dataset3(data.Dataset):
    def __init__(self, data_dir, file_list, labels_dict, device):
        super().__init__()
        self.device = device
        self.data_dir_axial = f"{data_dir}/axial"
        self.data_dir_coronal = f"{data_dir}/coronal"
        self.data_dir_sagittal = f"{data_dir}/sagittal"

        self.paths_axial = [os.path.join(self.data_dir_axial, file) for file in file_list]
        self.paths_coronal = [os.path.join(self.data_dir_coronal, file) for file in file_list]
        self.paths_sagittal = [os.path.join(self.data_dir_sagittal, file) for file in file_list]
        
        self.paths = [self.paths_axial, self.paths_coronal, self.paths_sagittal]
        
        self.labels = [labels_dict[file] for file in file_list]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy).to(self.device)
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=weights_tensor)
        return loss

    def __getitem__(self, index):
        vol_list = []

        for i in range(3):           
            path = self.paths[i][index]
            vol = np.load(path).astype(np.int32)

            pad = int((vol.shape[2] - INPUT_DIM) / 2)
            vol = vol[:, pad:-pad, pad:-pad]

            vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
            vol = (vol - MEAN) / STDDEV
            vol = np.stack((vol,) * 3, axis=1)

            vol_tensor = torch.FloatTensor(vol).to(self.device)


            vol_list.append(vol_tensor)

        
        label_tensor = torch.FloatTensor([self.labels[index]]).to(self.device)

        return vol_list, label_tensor

    def __len__(self):
        return len(self.labels)


class Dataset3(data.Dataset):
    def __init__(self, data_dir, file_list, labels_dict, device):
        super().__init__()
        self.device = device
        self.data_dir_axial = f"{data_dir}/axial"
        self.data_dir_coronal = f"{data_dir}/coronal"
        self.data_dir_sagittal = f"{data_dir}/sagittal"

        self.paths_axial = [os.path.join(self.data_dir_axial, file) for file in file_list]
        self.paths_coronal = [os.path.join(self.data_dir_coronal, file) for file in file_list]
        self.paths_sagittal = [os.path.join(self.data_dir_sagittal, file) for file in file_list]
        
        self.paths = [self.paths_axial, self.paths_coronal, self.paths_sagittal]
        
        self.labels = [labels_dict[file] for file in file_list]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

        # Define the ResNet transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),           # Resize to 256x256
            transforms.CenterCrop(224),       # Center crop to 224x224
            transforms.ToTensor(),            # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy).to(self.device)
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=weights_tensor)
        return loss

    def __getitem__(self, index):
        vol_list = []

        for i in range(3):  # Loop over axial, coronal, sagittal planes
            path = self.paths[i][index]
            vol = np.load(path).astype(np.int32)

            pad = int((vol.shape[2] - INPUT_DIM) / 2)
            vol = vol[:, pad:-pad, pad:-pad]

            vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

            # Convert volume to PIL image format for transformations
            vol = Image.fromarray(vol[0])  # Select the first slice for demonstration
            
            # Apply transformations compatible with ResNet
            vol_tensor = self.transform(vol)

            vol_list.append(vol_tensor)

        # Stack the transformed volumes along the channel dimension
        vol_tensor = torch.stack(vol_list, dim=0).to(self.device)

        label_tensor = torch.FloatTensor([self.labels[index]]).to(self.device)

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.labels)


    train_dataset = Dataset3(data_dir, train_files, labels_dict, device)
    valid_dataset = Dataset3(data_dir, valid_files, labels_dict, device)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)

    return train_loader, valid_loader