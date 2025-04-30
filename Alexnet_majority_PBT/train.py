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
    def __init__(self, hyperparams, device, data_dir, labels_csv, batch_size):
        self.hyperparams = hyperparams
        self.device = device
        self.batch_size = batch_size

        self.model = MRNet3(dropout=hyperparams['dropout'])
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


def mutate_hyperparams(hparams):
    def mutate(value, factor_range=(0.8, 1.2), bounds=(1e-6, 1.0)):
        factor = np.random.uniform(*factor_range)
        new_value = value * factor
        return max(min(new_value, bounds[1]), bounds[0])

    return {
        "learning_rate": mutate(hparams["learning_rate"], bounds=(1e-5, 1e-1)),
        "weight_decay": mutate(hparams["weight_decay"], bounds=(1e-6, 1e-2)),
        "dropout": np.clip(hparams["dropout"] + np.random.uniform(-0.05, 0.05), 0.1, 0.7)
    }


def exploit_and_explore(population):
    population.sort(key=lambda m: m.best_val_auc, reverse=True)
    top_model = population[0]
    worst_model = population[-1]

    if worst_model.best_val_auc < top_model.best_val_auc:
        worst_model.model.load_state_dict(copy.deepcopy(top_model.checkpoint))
        new_hparams = mutate_hyperparams(top_model.hyperparams)
        worst_model.__init__(new_hparams, worst_model.device,
                             worst_model.train_loader.dataset.data_dir,
                             worst_model.train_loader.dataset.labels_csv,
                             worst_model.batch_size)


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
    for _ in range(args.population):
        init_hparams = {
            "learning_rate": np.random.uniform(1e-4, 1e-2),
            "weight_decay": np.random.uniform(1e-5, 1e-3),
            "dropout": np.random.uniform(0.3, 0.6)
        }
        member = PBTMember(init_hparams, device, args.data_dir, args.labels_csv, args.batch_size)
        population.append(member)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        epoch_logs = []

        for i, member in enumerate(population):
            train_loss, train_auc, _, _ = run_model(member.model, member.train_loader, train=True, optimizer=member.optimizer, eps=args.eps)
            val_loss, val_auc, _, _ = run_model(member.model, member.val_loader, train=False)
            member.scheduler.step(val_loss)

            if val_auc > member.best_val_auc:
                member.best_val_auc = val_auc
                member.checkpoint = copy.deepcopy(member.model.state_dict())

            log_entry = {
                "epoch": epoch + 1,
                "model_id": i,
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

        # Print only the best model AUCs
        best_model = max(epoch_logs, key=lambda x: x['val_auc'])
        print(f"Best Model {best_model['model_id']}: train_auc={best_model['train_auc']:.4f}, val_auc={best_model['val_auc']:.4f}")

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
