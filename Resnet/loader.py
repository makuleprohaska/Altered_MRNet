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

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

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

        # Define transformation pipeline for ResNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.ToTensor(),   # Convert PIL Image to tensor, scales to [0, 1]
            transforms.Normalize(mean=MEAN, std=STDDEV)  # Normalize for ResNet
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
            vol = np.load(path).astype(np.float32)  # Load as float32

            # Crop to INPUT_DIM x INPUT_DIM
            pad = int((vol.shape[2] - INPUT_DIM) / 2)
            vol = vol[:, pad:-pad, pad:-pad]

            # Normalize to [0, 1]
            vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))

            # Convert to 3 channels (repeat grayscale)
            vol = np.stack([vol] * 3, axis=1)  # Shape: [num_slices, 3, H, W]

            # Apply transformation to each slice
            vol_slices = []
            for slice_idx in range(vol.shape[0]):
                slice_np = vol[slice_idx].transpose(1, 2, 0)  # Shape: [H, W, 3]
                slice_np = (slice_np * 255).astype(np.uint8)  # Scale to [0, 255] for PIL
                slice_tensor = self.transform(slice_np)  # Shape: [3, H, W]
                vol_slices.append(slice_tensor)
            
            # Stack slices into a tensor
            vol_tensor = torch.stack(vol_slices).to(self.device)  # Shape: [num_slices, 3, H, W]
            vol_list.append(vol_tensor)

        label_tensor = torch.FloatTensor([self.labels[index]]).to(self.device)

        return vol_list, label_tensor

    def __len__(self):
        return len(self.labels)

def load_data3(device, data_dir, labels_csv):
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    all_files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith(".npy")]
    all_files = [f for f in all_files if f in labels_dict]
    all_files.sort()

    labels = [labels_dict[file] for file in all_files]
    train_files, valid_files = train_test_split(
        all_files, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )

    train_dataset = Dataset3(data_dir, train_files, labels_dict, device)
    valid_dataset = Dataset3(data_dir, valid_files, labels_dict, device)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=0, shuffle=False)

    return train_loader, valid_loader