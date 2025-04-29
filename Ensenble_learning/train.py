import argparse
import json
import numpy as np
import os
import torch
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from model import EnsembleModel
from loader import load_data, load_data_test

def get_device(use_gpu, use_mps):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def run_model(model, loader, train=False, optimizer=None):
    from sklearn import metrics
    from tqdm import tqdm
    preds, labels = [], []
    total_loss = 0.0
    num_batches = 0

    if train:
        model.train()
    else:
        model.eval()

    for batch in tqdm(loader, desc="Processing batches", total=len(loader)):
        if train:
            optimizer.zero_grad()
        vol, label, original_slices = batch
        label = label.to(loader.dataset.device)

        with torch.no_grad() if not train else torch.enable_grad():
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

def train_ensemble(rundir, epochs, learning_rate, use_gpu, use_mps, data_dir, labels_csv, 
                   alexnet_model_path, resnet_model_path, weight_decay, max_patience, batch_size, label_smoothing):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")

    # Load data
    labels_df = pd.read_csv(labels_csv, header=None, names=['filename', 'label'])
    labels_df['filename'] = labels_df['filename'].apply(lambda x: f"{int(x):04d}.npy")
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))
    all_files = [f for f in os.listdir(f"{data_dir}/axial") if f.endswith(".npy") and f in labels_dict]
    all_files.sort()
    labels = [labels_dict[file] for file in all_files]
    train_files, valid_files = train_test_split(all_files, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = MRDataset(data_dir, train_files, labels_dict, device, train=True, label_smoothing=label_smoothing)
    valid_dataset = MRDataset(data_dir, valid_files, labels_dict, device, train=False, label_smoothing=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = EnsembleModel(alexnet_model_path, resnet_model_path).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max_patience, factor=0.3)

    best_val_auc = float('-inf')
    start_time = datetime.now()

    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}. Time passed: {datetime.now() - start_time}')
        
        # Training
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'Train loss: {train_loss:.4f}, AUC: {train_auc:.4f}')

        # Validation
        val_loss, val_auc, _, _ = run_model(model, valid_loader, train=False)
        print(f'Valid loss: {val_loss:.4f}, AUC: {val_auc:.4f}')

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            file_name = f'val{val_auc:.4f}_train{train_auc:.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            print(f"Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--alexnet_model_path', type=str, required=True)
    parser.add_argument('--resnet_model_path', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.makedirs(args.rundir, exist_ok=True)
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train_ensemble(args.rundir, args.epochs, args.learning_rate, args.gpu, args.mps, args.data_dir, 
                   args.labels_csv, args.alexnet_model_path, args.resnet_model_path, args.weight_decay, 
                   args.max_patience, args.batch_size, args.label_smoothing)