import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm

from loader import load_data3
from loader import load_data_test
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

def run_model(model, loader, train=False, optimizer=None, accumulation_steps=4):
    """
    Run the model on the given data loader with gradient accumulation for training.
    
    Args:
        model: The neural network model.
        loader: DataLoader providing the batches.
        train: Boolean indicating training or evaluation mode.
        optimizer: Optimizer used for weight updates (required if train=True).
        accumulation_steps: Number of batches to accumulate gradients over (default: 4).
    
    Returns:
        avg_loss: Average loss over the dataset.
        auc: Area under the ROC curve.
        preds: List of predictions.
        labels: List of true labels.
    """
    preds = []
    labels = []
    total_loss = 0.
    num_batches = 0

    # Set model mode
    if train:
        model.train()
    else:
        model.eval()

    # Process batches
    for i, batch in enumerate(tqdm(loader, desc="Processing batches", total=len(loader))):
        vol, label, original_slices = batch
        vol_device = vol  # Assuming vol is already on the correct device
        label = label.to(loader.dataset.device)

        # Zero gradients at the start of an accumulation cycle
        if train and i % accumulation_steps == 0:
            optimizer.zero_grad()

        # Forward pass
        if str(loader.dataset.device).startswith('cuda'):
            with autocast(enabled=True):  # Mixed precision for CUDA
                logit = model.forward(vol_device, original_slices)
                loss = loader.dataset.weighted_loss(logit, label, train) / accumulation_steps
        else:
            logit = model.forward(vol_device, original_slices)
            loss = loader.dataset.weighted_loss(logit, label, train) / accumulation_steps

        # Backward pass for training
        if train:
            loss.backward()  # Accumulate gradients

            # Update weights after accumulation_steps batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Track loss (scale back for logging)
        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # Collect predictions and labels
        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy().flatten()
        label_npy = label.data.cpu().numpy().flatten()
        preds.extend(pred_npy)
        labels.extend(label_npy)

    # Handle remaining gradients at the end of the epoch
    if train and num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Compute average loss and AUC
    avg_loss = total_loss / num_batches
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels

def evaluate(split, model_path, use_gpu, use_mps, data_dir, labels_csv, batch_size, label_smoothing):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")
    
    if split == 'train' or split == 'valid':
        train_loader, valid_loader = load_data3(device, data_dir, labels_csv, batch_size=batch_size, label_smoothing=label_smoothing)

    elif split == 'test':
        test_loader = load_data_test(device, data_dir, labels_csv, batch_size=batch_size, label_smoothing=label_smoothing)

    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")
    
    print("Loading model from path:", model_path)

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

    loss, auc, preds, labels = run_model(model, loader, train=False)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.gpu, args.mps, args.data_dir, args.labels_csv, args.batch_size, args.label_smoothing)