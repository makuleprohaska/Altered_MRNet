import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from loader import load_data3  # Updated load_data3 with 260x260 inputs
from model import MRNet3  # Updated MRNet3 with EfficientNet-B2

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
    parser.add_argument('--split', type=str, required=True, choices=['train', 'valid', 'test'])
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV file')
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--mps', action='store_true', help='Use MPS if available')
    return parser

def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    num_batches = 0
    print(f"Number of batches: {len(loader)}")
    
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="Processing batches", total=len(loader)):
            if train and optimizer:
                optimizer.zero_grad()

            # batch[0] is a list of 3 tensors [axial, coronal, sagittal]
            vol, label = batch[0], batch[1]
            
            # Move volume and label to the correct device
            vol_device = [i.to(loader.dataset.device) for i in vol]
            label = label.to(loader.dataset.device)

            logit = model(vol_device)  # Single logit output
            loss = loader.dataset.weighted_loss(logit, label)
            total_loss += loss.item()

            pred = torch.sigmoid(logit)
            pred_npy = pred.cpu().numpy().item()  # Scalar for single batch
            label_npy = label.cpu().numpy().item()

            preds.append(pred_npy)
            labels.append(label_npy)

            if train and optimizer:
                loss.backward()
                optimizer.step()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Compute AUC, handling edge case of uniform labels
    try:
        fpr, tpr, _ = metrics.roc_curve(labels, preds)
        auc = metrics.auc(fpr, tpr)
    except ValueError:
        auc = float('nan')  # If labels are all 0s or 1s

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, diagnosis, use_gpu, use_mps, data_dir, labels_csv):
    # Set up the device
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")
    
    # Load data with updated load_data3 (now returns train, valid, test)
    train_loader, valid_loader, test_loader = load_data3(device, data_dir, labels_csv, diagnosis)

    # Initialize the model (MRNet3 with EfficientNet-B2)
    model = MRNet3()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Choose the loader based on split
    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    # Run evaluation
    loss, auc, preds, labels = run_model(model, loader, train=False)

    print(f'{split} loss: {loss:.4f}')
    print(f'{split} AUC: {auc:.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu, args.mps, args.data_dir, args.labels_csv)