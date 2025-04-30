import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm
from loader import load_data3
from model import MRNet3
from torch.cuda.amp import autocast
 



def get_device(use_gpu, use_mps):
    
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    
    elif use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    
    else:
        return torch.device("cpu")


def get_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV file')
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--mps', action='store_true', help='Use MPS if available')
    
    return parser

def run_model(model, loader, train=False, optimizer=None, eps: float = 0.0):
    """
    model    : your MRNet3 instance
    loader   : DataLoader returning (vol_lists, label)
    train    : whether to do optimizer.step()
    optimizer: your Adam optimizer (only used if train=True)
    eps      : label-smoothing factor (0.0 = no smoothing)
    """
    preds = []
    labels = []
    total_loss = 0.0
    num_batches = 0

    if train:
        model.train()
    else:
        model.eval()

    device = loader.dataset.device

    for vol_lists, label in tqdm(loader, desc="Processing batches", total=len(loader)):
        # Move data to device
        label = label.to(device)                       # [batch_size,1]
        vol_lists = [[view.to(device) for view in views] for views in vol_lists]

        with torch.autocast(device_type=device.type, enabled=True):
            logits = model(vol_lists)                      # [batch_size,1]
        probs  = torch.sigmoid(logits)                 # [batch_size,1]

        # Loss
        if train:
            # use smoothing in training
            loss = loader.dataset.weighted_loss(logits, label, eps=eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # no smoothing in val/test
            loss = loader.dataset.weighted_loss(logits, label, eps=0.0)

        # Accumulate
        total_loss += loss.item()
        preds.extend(probs.detach().cpu().view(-1).tolist())
        labels.extend(label.detach().cpu().view(-1).tolist())
        num_batches += 1

    avg_loss = total_loss / num_batches
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels
# def run_model(model, loader, train=False, optimizer=None):
    
#     preds = []
#     labels = []
    
#     if train:
#         model.train()
    
#     else:
#         model.eval()
    
#     total_loss = 0.
#     num_batches = 0
    
#     print(f"num_batches: {len(loader)}")
    
#     for batch in tqdm(loader, desc="Processing batches", total=len(loader)):
        
#         if train:
#             optimizer.zero_grad()
        
#         vol_lists, label = batch  # vol_lists: [[axial, coronal, sagittal], ...]
#         label = label.to(loader.dataset.device)  # [batch_size, 1]
        
#         # Move all tensors to device
#         vol_lists_device = [[view.to(loader.dataset.device) for view in vol_list] for vol_list in vol_lists]
        
#         logit = model.forward(vol_lists_device)  # [batch_size, 1]
        
#         loss = loader.dataset.weighted_loss(logit, label)
#         total_loss += loss.item()
        
#         pred = torch.sigmoid(logit)
#         pred_npy = pred.data.cpu().numpy()[:, 0]  # (batch_size,)
#         label_npy = label.data.cpu().numpy()[:, 0]  # (batch_size,)
#         preds.extend(pred_npy.tolist())
#         labels.extend(label_npy.tolist())
        
#         if train:
#             loss.backward()
#             optimizer.step()
#         num_batches += 1
    
#     avg_loss = total_loss / num_batches
#     fpr, tpr, threshold = metrics.roc_curve(labels, preds)
#     auc = metrics.auc(fpr, tpr)
    
#     return avg_loss, auc, preds, labels


def evaluate(split, model_path, diagnosis, use_gpu, mps, data_dir, labels_csv):
    device = get_device(use_gpu, mps)
    print(f"Using device: {device}")
    train_loader, valid_loader, test_loader = load_data3(device, data_dir, labels_csv, diagnosis)
    model = MRNet3()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")
    loss, auc, preds, labels = run_model(model, loader, train=False)
    print(f'{split} loss: {loss:.4f}')
    print(f'{split} AUC: {auc:.4f}')
    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu, args.mps, args.data_dir, args.labels_csv)