import argparse
import os
import json
import copy
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from evaluate import run_model
from loader import load_data3
from model import MRNet3


class PBTMember:
    def __init__(self, id, hyperparams, device, data_dir, labels_csv, batch_size):
        self.id = id
        self.hyperparams = hyperparams
        self.device = device
        self.batch_size = batch_size

        use_batchnorm = batch_size > 1
        self.model = MRNet3(dropout=hyperparams['dropout'], use_batchnorm=use_batchnorm)
        self.model.to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.3
        )

        self.train_loader, self.val_loader = load_data3(device, data_dir, labels_csv, batch_size=batch_size)
        self.best_val_auc = 0.0
        self.checkpoint = None

        #log mutations
        self.copied_from = None


def mutate_hyperparams(hparams):
    def mutate(value, factor_range=(0.7, 1.3), bounds=(1e-6, 1.0)):
        factor = np.random.uniform(*factor_range)
        new_value = value * factor
        return max(min(new_value, bounds[1]), bounds[0])

    return {
        "learning_rate": mutate(hparams["learning_rate"], bounds=(1e-6, 1e-3)),
        "weight_decay": mutate(hparams["weight_decay"], bounds=(1e-6, 1e-2)),
        "dropout": np.clip(hparams["dropout"] + np.random.uniform(-0.1, 0.1), 0.05, 0.7)
    }


def exploit_and_explore(population):
    import numpy as np
    # sort so best‐performers are first
    population.sort(key=lambda m: m.best_val_auc, reverse=True)
    top_k    = population[:2]   # two strongest
    bottom_k = population[-1:]  # only the single weakest

    # build a softmax over the top2’s AUCs
    aucs  = np.array([m.best_val_auc for m in top_k])
    probs = np.exp(aucs) / np.sum(np.exp(aucs))

    for target in bottom_k:
        parent = np.random.choice(top_k, p=probs)
        if target.best_val_auc < parent.best_val_auc:
            # 1) copy _only_ the weights
            target.model.load_state_dict(copy.deepcopy(parent.checkpoint))

            # Log lineage
            target.copied_from = parent.id
            
            # 2) compute new hyperparams
            new_h = mutate_hyperparams(parent.hyperparams)
            target.hyperparams = new_h

            # 3) update optimizer _in place_
            for pg in target.optimizer.param_groups:
                pg['lr']           = new_h['learning_rate']
                pg['weight_decay'] = new_h['weight_decay']

            # 4) update dropout _in place_
            target.model.update_dropout(new_h['dropout'])


def pbt_train(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else
                          "mps" if args.mps and torch.backends.mps.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)
    log_path = Path(args.rundir) / "population_log.jsonl"

    population = []
    for i in range(args.population):
        init_hparams = {
            "learning_rate": np.random.uniform(0.5e-5, 2e-5),
            "weight_decay": np.random.uniform(0.001, 0.004),
            "dropout": np.random.uniform(0.1, 0.3)
        }
        member = PBTMember(i, init_hparams, device, args.data_dir, args.labels_csv, args.batch_size)
        population.append(member)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        epoch_logs = []

        for member in population:

            train_loss, train_auc, _, _ = run_model(member.model, member.train_loader, train=True, optimizer=member.optimizer, eps=args.eps)
            val_loss, val_auc, _, _ = run_model(member.model, member.val_loader, train=False)
            member.scheduler.step(val_loss)

            if val_auc > member.best_val_auc:
                member.best_val_auc = val_auc
                member.checkpoint = copy.deepcopy(member.model.state_dict())

            log_entry = {
                "epoch": epoch + 1,
                "copied_from": getattr(member, 'copied_from', None),
                "model_id": member.id,
                "learning_rate": member.hyperparams["learning_rate"],
                "weight_decay": member.hyperparams["weight_decay"],
                "dropout": member.hyperparams["dropout"],
                "train_auc": train_auc,
                "val_auc": val_auc
            }
            epoch_logs.append(log_entry)

        # Save logs to JSONL
        with open(log_path, 'a') as f:
            for entry in epoch_logs:
                f.write(json.dumps(entry) + '\n')

        # Clear lineage flags so they don't persist beyond this epoch
        for member in population:
            if hasattr(member, 'copied_from'):
                del member.copied_from

        # Print only the best model AUCs
        best_model = max(epoch_logs, key=lambda x: x['val_auc'])
        print(f"Best Model {best_model['model_id']}: train_auc={best_model['train_auc']:.4f}, val_auc={best_model['val_auc']:.4f}")

        # Exploit and explore after interval
        if (epoch + 1) % args.eval_interval == 0:
            exploit_and_explore(population)

    # Save final best model in population
    best_model = max(population, key=lambda m: m.best_val_auc)
    torch.save(best_model.checkpoint, Path(args.rundir) / 'best_model.pth')
    with open(Path(args.rundir) / 'best_hyperparams.json', 'w') as f:
        json.dump(best_model.hyperparams, f, indent=2)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--population', type=int, default=3)
    parser.add_argument('--eval_interval', type=int, default=5)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    pbt_train(args)
