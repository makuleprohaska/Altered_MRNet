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
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV file')
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--mps', action='store_true', help='Use MPS if available')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for evaluation')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing factor')
    return parser

def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0
    print(f"num_batches: {len(loader)}")
    for batch in tqdm(loader, desc="Processing batches", total=len(loader)):
        if train:
            optimizer.zero_grad()

        vol, label, original_slices = batch
        
        vol_device = vol  # List of [B, S_max, 3, 224, 224]
        label = label.to(loader.dataset.device)

        if str(loader.dataset.device).startswith('cuda'):
            with autocast(enabled=True):
                logit = model.forward(vol_device, original_slices)
                loss = loader.dataset.weighted_loss(logit, label, train)
        else:
            logit = model.forward(vol_device, original_slices)
            loss = loader.dataset.weighted_loss(logit, label, train)
        
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy().flatten()
        label_npy = label.data.cpu().numpy().flatten()

        preds.extend(pred_npy)
        labels.extend(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, use_gpu, use_mps, data_dir, labels_csv, batch_size, label_smoothing):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")
    
    train_loader, valid_loader = load_data3(device, data_dir, labels_csv, batch_size=batch_size, label_smoothing=label_smoothing)

    model = MRNet3()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    else:
        raise ValueError("split must be 'train' or 'valid'")

    loss, auc, preds, labels = run_model(model, loader, train=False)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.gpu, args.mps, args.data_dir, args.labels_csv, args.batch_size, args.label_smoothing)