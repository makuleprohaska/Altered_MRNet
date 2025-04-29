import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIM_227 = 227  # For AlexNet
INPUT_DIM_224 = 224  # For ResNet18
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

class MRDataset(data.Dataset):
    def __init__(self, data_dir, file_list, labels_dict, device, train=False, label_smoothing=0.1, augment=False):
        super().__init__()
        self.device = device
        self.train = train
        self.augment = augment
        self.data_dir_axial = f"{data_dir}/axial"
        self.data_dir_coronal = f"{data_dir}/coronal"
        self.data_dir_sagittal = f"{data_dir}/sagittal"

        self.paths_axial = [os.path.join(self.data_dir_axial, file) for file in file_list]
        self.paths_coronal = [os.path.join(self.data_dir_coronal, file) for file in file_list]
        self.paths_sagittal = [os.path.join(self.data_dir_sagittal, file) for file in file_list]
        
        self.paths = [self.paths_axial, self.paths_coronal, self.paths_sagittal]
        
        self.labels = [labels_dict[file] for file in file_list]
        self.label_smoothing = label_smoothing

        neg_weight = np.mean(self.labels)
        dtype = np.float32
        self.weights = [dtype(neg_weight), dtype(1 - neg_weight)]

    def weighted_loss(self, prediction, target, train):
        dtype = torch.float32
        indices = target.squeeze(1).long()
        weights_tensor = torch.tensor(self.weights, device=self.device, dtype=dtype)[indices].unsqueeze(1)
        smoothed_target = target * (1 - self.label_smoothing) + (1 - target) * self.label_smoothing if train and self.label_smoothing > 0 else target
        return F.binary_cross_entropy_with_logits(prediction, smoothed_target, weight=weights_tensor)

    def __getitem__(self, index):
        vol_list = []

        for i in range(3):  # Axial, Coronal, Sagittal
            vol = np.load(self.paths[i][index]).astype(np.float32)

            # Crop to 227x227 for AlexNet
            pad_227 = int((vol.shape[2] - INPUT_DIM_227) / 2)
            vol_227 = vol[:, pad_227:pad_227 + INPUT_DIM_227, pad_227:pad_227 + INPUT_DIM_227]
            vol_227 = (vol_227 - np.min(vol_227)) / (np.max(vol_227) - np.min(vol_227) + 1e-6)
            vol_227 = np.stack((vol_227,) * 3, axis=1)  # [slices, 3, 227, 227]
            vol_227_tensor = torch.FloatTensor(vol_227).to(self.device)
            for c in range(3):
                vol_227_tensor[:, c, :, :] = (vol_227_tensor[:, c, :, :] - MEAN[c]) / STDDEV[c]
            vol_list.append(vol_227_tensor)

            # Crop to 224x224 for ResNet18
            pad_224 = int((vol.shape[2] - INPUT_DIM_224) / 2)
            vol_224 = vol[:, pad_224:pad_224 + INPUT_DIM_224, pad_224:pad_224 + INPUT_DIM_224]
            vol_224 = (vol_224 - np.min(vol_224)) / (np.max(vol_224) - np.min(vol_224) + 1e-6)
            vol_224 = np.stack((vol_224,) * 3, axis=1)  # [slices, 3, 224, 224]
            vol_224_tensor = torch.FloatTensor(vol_224).to(self.device)
            for c in range(3):
                vol_224_tensor[:, c, :, :] = (vol_224_tensor[:, c, :, :] - MEAN[c]) / STDDEV[c]
            vol_list.append(vol_224_tensor)

        label_tensor = torch.FloatTensor([self.labels[index]]).to(self.device)
        return vol_list, label_tensor

    def __len__(self):
        return len(self.labels)

def collate_fn(batch):
    device = batch[0][0][0].device
    vol_227_axial_list = [sample[0][0] for sample in batch]
    vol_227_coronal_list = [sample[0][1] for sample in batch]
    vol_227_sagittal_list = [sample[0][2] for sample in batch]
    vol_224_axial_list = [sample[0][3] for sample in batch]
    vol_224_coronal_list = [sample[0][4] for sample in batch]
    vol_224_sagittal_list = [sample[0][5] for sample in batch]

    padded_vol_227_axial = torch.nn.utils.rnn.pad_sequence(vol_227_axial_list, batch_first=True)
    padded_vol_227_coronal = torch.nn.utils.rnn.pad_sequence(vol_227_coronal_list, batch_first=True)
    padded_vol_227_sagittal = torch.nn.utils.rnn.pad_sequence(vol_227_sagittal_list, batch_first=True)
    padded_vol_224_axial = torch.nn.utils.rnn.pad_sequence(vol_224_axial_list, batch_first=True)
    padded_vol_224_coronal = torch.nn.utils.rnn.pad_sequence(vol_224_coronal_list, batch_first=True)
    padded_vol_224_sagittal = torch.nn.utils.rnn.pad_sequence(vol_224_sagittal_list, batch_first=True)

    original_slices_axial = torch.tensor([v.shape[0] for v in vol_227_axial_list], device=device)
    original_slices_coronal = torch.tensor([v.shape[0] for v in vol_227_coronal_list], device=device)
    original_slices_sagittal = torch.tensor([v.shape[0] for v in vol_227_sagittal_list], device=device)

    labels = torch.stack([sample[1] for sample in batch])
    vol_padded = [
        padded_vol_227_axial, padded_vol_227_coronal, padded_vol_227_sagittal,
        padded_vol_224_axial, padded_vol_224_coronal, padded_vol_224_sagittal
    ]
    return vol_padded, labels, [original_slices_axial, original_slices_coronal, original_slices_sagittal]

def load_data(device, data_dir, labels_csv, batch_size=1, label_smoothing=0.1, augment=False):
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    all_files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith(".npy") and f in labels_dict]
    all_files.sort()

    labels = [labels_dict[file] for file in all_files]
    train_files, valid_files = train_test_split(
        all_files, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = MRDataset(data_dir, train_files, labels_dict, device, train=True, label_smoothing=label_smoothing, augment=augment)
    valid_dataset = MRDataset(data_dir, valid_files, labels_dict, device, train=False, label_smoothing=0)

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, valid_loader

def load_data_test(device, data_dir, labels_csv, batch_size=1, label_smoothing=0):
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    test_files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith(".npy") and f in labels_dict]
    test_files.sort()

    test_dataset = MRDataset(data_dir, test_files, labels_dict, device, train=False, label_smoothing=label_smoothing)

    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False, 
        pin_memory=device.type == 'cuda', collate_fn=collate_fn
    )

    return test_loader