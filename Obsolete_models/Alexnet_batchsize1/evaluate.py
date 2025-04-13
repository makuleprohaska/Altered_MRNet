import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm

#from loader import load_data3
#from model import MRNet

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

        vol, label = batch[0], batch[1]
        

        vol_device = []
        for i in vol:
            i = i.to(loader.dataset.device)
            vol_device.append(i)
        label = label.to(loader.dataset.device)

        logit = model.forward(vol_device)
        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels

#TO BE MODIFIED TO USE THE MRNET3 MODEL

# def evaluate(split, model_path, diagnosis, use_gpu, use_mps, data_dir, labels_csv):
#     device = get_device(use_gpu, use_mps)
#     print(f"Using device: {device}")
#     train_loader, valid_loader, test_loader = load_data(diagnosis, device, data_dir, labels_csv)

#     model = MRNet()
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)

#     model = model.to(device)

#     if split == 'train':
#         loader = train_loader
#     elif split == 'valid':
#         loader = valid_loader
#     elif split == 'test':
#         loader = test_loader
#     else:
#         raise ValueError("split must be 'train', 'valid', or 'test'")

#     loss, auc, preds, labels = run_model(model, loader)

#     print(f'{split} loss: {loss:0.4f}')
#     print(f'{split} AUC: {auc:0.4f}')

#     return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu, args.mps, args.data_dir, args.labels_csv)


