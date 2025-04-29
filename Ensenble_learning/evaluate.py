import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
from model import EnsembleModel
from loader import load_data, load_data_test
from torch.cuda.amp import autocast

def get_device(use_gpu, use_mps):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def run_model(model, loader, train=False, optimizer=None):
    preds, labels = [], []
    total_loss = 0.0
    num_batches = 0
    print(f"num_batches: {len(loader)}")

    if train:
        model.train()
    else:
        model.eval()

    for batch in tqdm(loader, desc="Processing batches", total=len(loader)):
        if train:
            optimizer.zero_grad()

        vol, label, original_slices = batch
        label = label.to(loader.dataset.device)

        if str(loader.dataset.device).startswith('cuda'):
            with autocast(enabled=True):
                logit = model(vol, original_slices)
                loss = loader.dataset.weighted_loss(logit, label, train)
        else:
            logit = model(vol, original_slices)
            loss = loader.dataset.weighted_loss(logit, label, train)

        total_loss += loss.item()
        pred = torch.sigmoid(logit)
        preds.extend(pred.cpu().numpy().flatten())
        labels.extend(label.cpu().numpy().flatten())

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches
    auc = metrics.roc_auc_score(labels, preds)
    return avg_loss, auc, preds, labels

def evaluate(split, model_path, alexnet_model_path, resnet_model_path, use_gpu, use_mps, 
             data_dir, labels_csv, batch_size, label_smoothing):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")

    if split in ['train', 'valid']:
        train_loader, valid_loader = load_data(device, data_dir, labels_csv, 
                                              batch_size=batch_size, label_smoothing=label_smoothing)
    elif split == 'test':
        test_loader = load_data_test(device, data_dir, labels_csv, 
                                     batch_size=batch_size, label_smoothing=label_smoothing)
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    print("Loading model from path:", model_path)
    model = EnsembleModel(alexnet_model_path, resnet_model_path)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    loader = train_loader if split == 'train' else valid_loader if split == 'valid' else test_loader
    loss, auc, preds, labels = run_model(model, loader, train=False)

    print(f'{split} loss: {loss:.4f}')
    print(f'{split} AUC: {auc:.4f}')
    return preds, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--alexnet_model_path', type=str, required=True)
    parser.add_argument('--resnet_model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    args = parser.parse_args()

    evaluate(args.split, args.model_path, args.alexnet_model_path, args.resnet_model_path, 
             args.gpu, args.mps, args.data_dir, args.labels_csv, args.batch_size, args.label_smoothing)