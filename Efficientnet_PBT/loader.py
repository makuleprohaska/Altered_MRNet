# loader.py
import os, random, numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import kornia.augmentation as K
from sklearn.model_selection import train_test_split

INPUT_DIM    = 224
MAX_PIXEL_VAL= 1.0
MEAN         = [0.485, 0.456, 0.406]
STDDEV       = [0.229, 0.224, 0.225]

class MRDataset(data.Dataset):
    def __init__(self, data_dir, file_list, labels_dict,
                 device, train=False, augment_prob=0.0):
        super().__init__()
        self.device       = device
        self.data_dir_ax  = f"{data_dir}/axial"
        self.data_dir_co  = f"{data_dir}/coronal"
        self.data_dir_sa  = f"{data_dir}/sagittal"
        self.paths_axial  = [os.path.join(self.data_dir_ax, file) for file in file_list]
        self.paths_coronal= [os.path.join(self.data_dir_co, file) for file in file_list]
        self.paths_sagittal=[os.path.join(self.data_dir_sa, file) for file in file_list]
        self.paths        = [self.paths_axial, self.paths_coronal, self.paths_sagittal]
        self.labels       = [labels_dict[f] for f in file_list]
        neg_weight       = np.mean(self.labels)
        self.weights     = [neg_weight, 1-neg_weight]
        self.train        = train
        self.augment_prob = augment_prob

    def weighted_loss(self, prediction, target, eps=0.0):
        # [B,1]
        tgt = target.view(-1,1)
        wts = np.array([self.weights[int(t.item())] for t in tgt.flatten()])
        wts = torch.FloatTensor(wts).view(-1,1).to(self.device)
        sm = tgt*(1-eps)+(1-tgt)*eps
        return F.binary_cross_entropy_with_logits(prediction, sm, weight=wts)

    def apply_augmentations(self, vol_tensor):
        vol_tensor = K.RandomRotation(degrees=25)(vol_tensor)
        vol_tensor = K.RandomAffine(degrees=0, translate=(25/224,25/224))(vol_tensor)
        if random.random()>0.5:
            vol_tensor = K.RandomHorizontalFlip(p=1.0)(vol_tensor)
        return vol_tensor

    def __getitem__(self, idx):
        vol_list = []
        for view_idx in range(3):
            path = self.paths[view_idx][idx]
            vol  = np.load(path).astype(np.float32)
            pad  = int((vol.shape[2]-INPUT_DIM)/2)
            vol  = vol[:,pad:-pad,pad:-pad]
            vol  = (vol - vol.min()) / (vol.max()-vol.min()) * MAX_PIXEL_VAL
            vol  = np.stack((vol,)*3, axis=1)  # (S,3,224,224)
            vol  = torch.FloatTensor(vol)
            # augmentation per sample
            if self.train and random.random() < self.augment_prob:
                vol = self.apply_augmentations(vol)
            vol_list.append(vol)
        label = torch.FloatTensor([self.labels[idx]])
        return vol_list, label

    def __len__(self):
        return len(self.labels)

def collate_fn(batch):
    # batch: list of ( [S_ax], [S_co], [S_sa] , label )
    device = batch[0][0][0].device
    views = list(zip(*[b[0] for b in batch]))  # 3 lists of tensors
    padded = []
    orig_s = []
    for view_list in views:
        # pad each view to max slices
        S = [v.shape[0] for v in view_list]
        orig_s.append(torch.tensor(S, device=device))
        padded.append(torch.nn.utils.rnn.pad_sequence(view_list, batch_first=True))
    labels = torch.stack([b[1] for b in batch], dim=0)
    return padded, labels, orig_s

def load_data3(device, data_dir, labels_csv,
               batch_size=1, augment_prob=0.0, collate_fn=None):
    df = pd.read_csv(labels_csv, header=None, names=['filename','label'])
    df.filename = df.filename.apply(lambda x:f"{int(x):04d}.npy")
    labels_dict = dict(zip(df.filename, df.label))
    files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith('.npy') and f in labels_dict]
    files.sort()
    labels = [labels_dict[f] for f in files]
    train_files, valid_files = train_test_split(files, test_size=0.2,
                                                random_state=42, stratify=labels)
    train_ds = MRDataset(data_dir, train_files, labels_dict,
                         device, train=True,  augment_prob=augment_prob)
    valid_ds = MRDataset(data_dir, valid_files, labels_dict,
                         device, train=False, augment_prob=0.0)
    tr = data.DataLoader(train_ds, batch_size=batch_size,
                         shuffle=True,  pin_memory=(device.type=='cuda'),
                         collate_fn=collate_fn)
    va = data.DataLoader(valid_ds, batch_size=batch_size,
                         shuffle=False, pin_memory=(device.type=='cuda'),
                         collate_fn=collate_fn)
    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}")
    return tr, va
