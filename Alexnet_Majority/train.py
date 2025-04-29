import argparse
import json
import numpy as np
import os
import torch
from datetime import datetime
from pathlib import Path
from sklearn import metrics
from evaluate import run_model
from loader import load_data3
from model import MRNet3

def get_device(use_gpu, use_mps):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train3(rundir, epochs, learning_rate, use_gpu, use_mps, data_dir, labels_csv, weight_decay, max_patience, batch_size, augment, epsilon):
    device = get_device(use_gpu, use_mps)
    print(f"Using device: {device}")
    train_loader, valid_loader = load_data3(device, data_dir, labels_csv, batch_size=batch_size, augment=augment)
    
    #This now deals with the case that batch size is 1
    use_batchnorm = batch_size > 1
    model = MRNet3(use_batchnorm=use_batchnorm)
    model = model.to(device)

    print(f"Using BatchNorm: {use_batchnorm}")

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max_patience, factor=.3, threshold=1e-4)

    best_val_auc = float('-inf')

    start_time = datetime.now()

    epsilon = args.eps
    print(f"Value of eps:{epsilon}")
    for epoch in range(epochs):
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer, eps=epsilon)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader, eps=0.0)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            file_name = f'val{val_auc:0.4f}_train{train_auc:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name 
            torch.save(model.state_dict(), save_path)

        # Log metrics to file
        with open(os.path.join(rundir, 'metrics.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_auc={train_auc:.4f}, val_auc={val_auc:.4f}\n")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to labels CSV file')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')
    parser.add_argument('--mps', action='store_true', help='Use MPS if available')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.0025, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training and validation')
    parser.add_argument('--eps', default=0.0, type=float, help='Label smoothing factor (0.0 = no smoothing)')
    parser.add_argument('--augment', default=True, action='store_true', help='Apply data augmentation during training')
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
           args.gpu, args.mps, args.data_dir, args.labels_csv, args.weight_decay, args.max_patience, args.batch_size, args.augment, args.eps)
    
#to run use
"""
python train.py --epochs 5 \
--data_dir /Users/matteobruno/Desktop/train \
--labels_csv /Users/matteobruno/Desktop/train/train-abnormal.csv \
--mps --rundir /Users/matteobruno/Desktop/runs
"""