import argparse
import torch
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
from loader import load_data3, collate_fn
from model import MRNet3

def get_device(use_gpu, use_mps):
    if use_gpu and torch.cuda.is_available(): return torch.device('cuda')
    if use_mps and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--split', choices=['train','valid','test'], required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--labels_csv', required=True)
    p.add_argument('--batch_size',    type=int,   default=4)
    p.add_argument('--aug_prob',      type=float, default=0.0)
    p.add_argument('--dropout',       type=float, default=0.0)
    p.add_argument('--gpu',           action='store_true')
    p.add_argument('--mps',           action='store_true')
    return p

def run_model(model, loader, train=False, optimizer=None, eps=0.0):
    preds, labels = [], []
    total_loss, nb = 0.0, 0
    model.train() if train else model.eval()
    device = loader.dataset.device

    for vol_lists, label, original_slices in tqdm(loader, total=len(loader)):
        # 1) move labels
        label = label.to(device)

        # 2) move each whole padded‐tensor to device, keep shape [B,S,3,224,224]
        vol_lists = [views.to(device) for views in vol_lists]

        # 3) also move the original_slices tensors to device (for masking)
        original_slices = [s.to(device) for s in original_slices]

        # 4) now forward takes a list of 3 tensors, not lists of lists
        logits = model(vol_lists, original_slices)
        probs  = torch.sigmoid(logits)
        # loss
        loss = loader.dataset.weighted_loss(logits, label, eps)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        nb += 1
        preds.extend(probs.detach().cpu().view(-1).tolist())
        labels.extend(label.detach().cpu().view(-1).tolist())

    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    return total_loss/nb, metrics.auc(fpr, tpr), preds, labels

if __name__ == '__main__':
    args   = get_parser().parse_args()
    device = get_device(args.gpu, args.mps)
    print(f'Using device: {device}')
    # load data
    train_loader, valid_loader = load_data3(
        device, args.data_dir, args.labels_csv,
        batch_size=args.batch_size,
        augment_prob=args.aug_prob,
        collate_fn=collate_fn
    )
    loader = {'train': train_loader, 'valid': valid_loader}[args.split]
    # build model
    model = MRNet3(dropout=args.dropout).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    # eval
    loss, auc, _, _ = run_model(model, loader, train=False)
    print(f'{args.split} loss={loss:.4f}, AUC={auc:.4f}')

