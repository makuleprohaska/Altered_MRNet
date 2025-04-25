import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

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
        # Ensure target is [batch_size, 1]
        target = target.view(-1, 1)

        # Compute weights for each sample
        weights_npy = np.array([self.weights[int(t.item())] for t in target.flatten()])

        # Reshape weights to [batch_size, 1] to match prediction and target
        weights_tensor = torch.FloatTensor(weights_npy).view(-1, 1).to(target.device)

        # Compute loss with weights reshaped to [batch_size, 1]
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
            vol_tensor = torch.FloatTensor(vol)  # Keep on CPU
            vol_list.append(vol_tensor)
        label_tensor = torch.FloatTensor([self.labels[index]])  # Shape: [1]
        return vol_list, label_tensor

    def __len__(self):
        return len(self.labels)

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable slice counts.
    Returns a list of view tensors and a stacked label tensor.
    """
    vol_lists = [item[0] for item in batch]  # List of [axial, coronal, sagittal] for each sample
    labels = torch.stack([item[1] for item in batch], dim=0)  # Stack labels: [batch_size, 1]
    return vol_lists, labels

def load_data3(device, data_dir, labels_csv, diagnosis=0, batch_size=1):
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    
    # Filter files that exist in all 3 views
    valid_files = []
    valid_labels = []
    for _, row in labels_df.iterrows():
        fname = row['filename']
        exists_all_views = all(os.path.exists(os.path.join(data_dir, view, fname)) for view in ['axial', 'coronal', 'sagittal'])
        if exists_all_views:
            valid_files.append(fname)
            valid_labels.append(row['label'])
    
    labels_dict = dict(zip(valid_files, valid_labels))

    # Stratify split
    train_files, valid_files = train_test_split(
        valid_files,
        test_size=0.2,
        random_state=42,
        stratify=valid_labels
    )

    train_dataset = Dataset3(data_dir, train_files, labels_dict, device)
    valid_dataset = Dataset3(data_dir, valid_files, labels_dict, device)

    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        shuffle=True, 
        pin_memory=device.type == 'cuda',
        collate_fn=custom_collate_fn
    )

    valid_loader = data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        shuffle=False, 
        pin_memory=device.type == 'cuda',
        collate_fn=custom_collate_fn
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}")
    return train_loader, valid_loader
