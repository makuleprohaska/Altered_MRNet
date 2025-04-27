import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import random
import kornia.augmentation as K

INPUT_DIM = 224
MAX_PIXEL_VAL = 1.0  # ResNet expects [0, 1] before channel-wise normalization
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean for ResNet (per channel)
STDDEV = [0.229, 0.224, 0.225]  # ImageNet std for ResNet (per channel)

class MRDataset(data.Dataset):
    def __init__(self, data_dir, file_list, labels_dict, device, train=False, label_smoothing=0.1):
        super().__init__()
        self.device = device
        self.train = train  # Flag to enable augmentations during training
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
        indices = target.squeeze(1).long()  # Shape: [B]
        weights_tensor = torch.tensor(self.weights, device=self.device, dtype=dtype)[indices]  # Shape: [B]
        weights_tensor = weights_tensor.unsqueeze(1)  # Shape: [B, 1]

        if train and self.label_smoothing > 0:
            smoothed_target = target * (1 - self.label_smoothing) + (1 - target) * self.label_smoothing
        else:
            smoothed_target = target

        loss = F.binary_cross_entropy_with_logits(prediction, smoothed_target, weight=weights_tensor)
        return loss

    def __getitem__(self, index):
        vol_list = []

        for i in range(3):           
            path = self.paths[i][index]
            vol = np.load(path).astype(np.float32) 

            # Crop to 224x224
            pad = int((vol.shape[2] - INPUT_DIM) / 2)
            vol = vol[:, pad:-pad, pad:-pad]

            # Normalize to [0, 1]
            vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-6)

            # Stack to 3 channels
            vol = np.stack((vol,) * 3, axis=1)  # Shape: (slices, 3, 224, 224)

            # Convert to tensor
            vol_tensor = torch.FloatTensor(vol).to(self.device)  # Shape: (slices, 3, 224, 224)

            # Apply augmentations only during training
            if self.train:
                vol_tensor = self.apply_augmentations(vol_tensor)

            # Apply ImageNet normalization per channel
            for c in range(3):
                vol_tensor[:, c, :, :] = (vol_tensor[:, c, :, :] - MEAN[c]) / STDDEV[c]

            vol_list.append(vol_tensor)

        label_tensor = torch.FloatTensor([self.labels[index]]).to(self.device)

        return vol_list, label_tensor

    def apply_augmentations(self, vol_tensor):
        # vol_tensor: (slices, 3, 224, 224)
        vol_tensor = K.RandomRotation(degrees=25)(vol_tensor)
        vol_tensor = K.RandomAffine(degrees=0, translate=(25/224, 25/224))(vol_tensor)
        if random.random() > 0.5:
            vol_tensor = K.RandomHorizontalFlip(p=1.0)(vol_tensor)
        return vol_tensor


    # def apply_augmentations(self, vol_tensor):
    #     """
    #     Apply random rotation, shift, and horizontal flip to the stack of slices.
    #     The same parameters are applied to all slices in the stack for consistency.
    #     """
    #     slices, c, h, w = vol_tensor.shape

    #     # Random rotation between -25 and 25 degrees
    #     angle = random.uniform(-25, 25)

    #     # Random shift between -25 and 25 pixels
    #     shift_h = random.randint(-25, 25)
    #     shift_w = random.randint(-25, 25)

    #     # Random horizontal flip with 50% probability
    #     flip = random.random() > 0.5

    #     # Apply transformations to each slice
    #     augmented_slices = []
    #     for i in range(slices):
    #         slice_tensor = vol_tensor[i]  # Shape: (3, 224, 224)

    #         # Rotate
    #         slice_tensor = TF.rotate(slice_tensor, angle)

    #         # Shift (translate)
    #         slice_tensor = TF.affine(slice_tensor, angle=0, translate=(shift_w, shift_h), scale=1.0, shear=0)

    #         # Horizontal flip
    #         if flip:
    #             slice_tensor = TF.hflip(slice_tensor)

    #         augmented_slices.append(slice_tensor)

    #     # Stack augmented slices back into a tensor
    #     return torch.stack(augmented_slices, dim=0)  # Shape: (slices, 3, 224, 224)

    def __len__(self):
        return len(self.labels)

def collate_fn(batch):
    device = batch[0][0][0].device
    view0_list = [sample[0][0] for sample in batch]  # Axial
    view1_list = [sample[0][1] for sample in batch]  # Coronal
    view2_list = [sample[0][2] for sample in batch]  # Sagittal
    
    # Pad slices to the maximum in the batch for each view
    padded_view0 = torch.nn.utils.rnn.pad_sequence(view0_list, batch_first=True)
    padded_view1 = torch.nn.utils.rnn.pad_sequence(view1_list, batch_first=True)
    padded_view2 = torch.nn.utils.rnn.pad_sequence(view2_list, batch_first=True)
    
    # Store original slice counts
    original_slices0 = torch.tensor([v.shape[0] for v in view0_list], device=device)
    original_slices1 = torch.tensor([v.shape[0] for v in view1_list], device=device)
    original_slices2 = torch.tensor([v.shape[0] for v in view2_list], device=device)
    
    # Stack labels
    labels = torch.stack([sample[1] for sample in batch])
    
    return [padded_view0, padded_view1, padded_view2], labels, [original_slices0, original_slices1, original_slices2]

def load_data3(device, data_dir, labels_csv, batch_size=1, label_smoothing=0.1):
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

    train_dataset = MRDataset(data_dir, train_files, labels_dict, device, train=True, label_smoothing=label_smoothing)
    valid_dataset = MRDataset(data_dir, valid_files, labels_dict, device, train=False, label_smoothing=label_smoothing)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader

def load_data_test(device, data_dir, labels_csv, batch_size=1, label_smoothing=0.1):
    
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    test_files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith(".npy")]
    test_files = [f for f in test_files if f in labels_dict]
    test_files.sort()

    labels = [labels_dict[file] for file in test_files]


    test_dataset = MRDataset(data_dir, test_files, labels_dict, device, train=True, label_smoothing=label_smoothing)

    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn)

    return test_loader