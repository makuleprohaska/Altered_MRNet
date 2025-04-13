import argparse
import json
import numpy as np
import os
import torch
from datetime import datetime
from pathlib import Path
from sklearn import metrics

from loader import load_data3  # Adapted dataset code with INPUT_DIM=260
from model import MRNet3  # Adapted MRNet3 with EfficientNet-B2

def get_device(use_gpu, use_mps):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def run_model(model, data_loader, train=False, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.set_grad_enabled(train):
        for data, target in data_loader:
            data = [d.to(device) for d in data]  # Move each view to device
            target = target.to(device)
            
            # Forward pass
            output = model(data)  # Single logit
            loss = data_loader.dataset.weighted_loss(output, target)
            
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Compute predictions for metrics
            pred_prob = torch.sigmoid(output).cpu().numpy()
            pred_label = (pred_prob >= 0.5).astype(int)
            
            all_preds.append(pred_prob.item())
            all_labels.append(target.cpu().numpy().item())
            
            correct += (pred_label == target.cpu().numpy()).sum()
            total += 1
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    auc = metrics.roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else float('nan')
    
    return avg_loss, auc, accuracy, all_preds

def train3(rundir, epochs, learning_rate, use_gpu, use_mps, data_dir, labels_csv):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")
    train_loader, valid_loader = load_data3(device, data_dir, labels_csv)
    
    model = MRNet3()
    model = model.to(device)

    # Adjusted learning rate and weight decay for EfficientNet-B2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.3, threshold=1e-4)

    best_val_loss = float('inf')
    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print(f'starting epoch {epoch+1}. time passed: {str(change)}')
        
        train_loss, train_auc, train_acc, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:.4f}')
        print(f'train AUC: {train_auc:.4f}')
        print(f'train accuracy: {train_acc:.4f}')

        val_loss, val_auc, val_acc, _ = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:.4f}')
        print(f'valid AUC: {val_auc:.4f}')
        print(f'valid accuracy: {val_acc:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            file_name = f'val{val_loss:.4f}_train{train_loss:.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            torch.save(model.state_dict(), save_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV file')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--mps', action='store_true', help='Use MPS if available')
    parser.add_argument('--learning_rate', default=3e-5, type=float)  # Lower for EfficientNet-B2
    parser.add_argument('--weight_decay', default=0.02, type=float)  # Slightly higher for deeper model
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    elif args.mps and torch.backends.mps.is_available():
        pass

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train3(args.rundir, args.epochs, args.learning_rate, 
           args.gpu, args.mps, args.data_dir, args.labels_csv)