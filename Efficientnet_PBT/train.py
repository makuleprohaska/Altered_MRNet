# train.py
import argparse, os, json, copy, random, numpy as np, torch
from pathlib import Path
from evaluate import run_model
from loader    import load_data3, collate_fn
from model     import MRNet3

class PBTMember:
    def __init__(self, id, hyperparams, device, data_dir, labels_csv):
        self.id         = id
        self.hyperparams= hyperparams
        self.device     = device
        # model + optimizer + scheduler
        self.model = MRNet3(dropout=hyperparams['dropout']).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.3
        )
        # data loaders driven by PBT hyperparams
        ap   = hyperparams['aug_prob']
        # always use batch_size=1
        self.train_loader, self.val_loader = load_data3(
            device, data_dir, labels_csv,
            batch_size=1,
            augment_prob=ap,
            collate_fn=collate_fn
        )
        self.best_val_auc = 0.0
        self.checkpoint   = None
        self.copied_from  = None

def mutate_hyperparams(h):
    def mval(v, f=(0.8,1.2), b=(1e-6,1.0)):
        nv = v * np.random.uniform(*f)
        return max(min(nv, b[1]), b[0])
    return {
        'learning_rate': mval(h['learning_rate'], b=(1e-6,1e-3)),
        'weight_decay' : mval(h['weight_decay'],  b=(1e-6,1e-2)),
        'dropout'      : float(np.clip(h['dropout'] + np.random.uniform(-0.05,0.05), 0.0,1.0)),
        'aug_prob'     : float(np.clip(h['aug_prob']  + np.random.uniform(-0.1,0.1),    0.0,1.0)),
        #batch size removed
    }

def exploit_and_explore(pop):
    pop.sort(key=lambda m: m.best_val_auc, reverse=True)
    top2    = pop[:2]
    worst   = pop[-1:]
    aucs    = np.array([m.best_val_auc for m in top2])
    probs   = np.exp(aucs)/np.sum(np.exp(aucs))
    for target in worst:
        parent = np.random.choice(top2, p=probs)
        if target.best_val_auc < parent.best_val_auc:
            target.model.load_state_dict(copy.deepcopy(parent.checkpoint))
            target.copied_from = parent.id
            new_h = mutate_hyperparams(parent.hyperparams)
            target.hyperparams = new_h
            # update optimizer
            for pg in target.optimizer.param_groups:
                pg['lr']           = new_h['learning_rate']
                pg['weight_decay'] = new_h['weight_decay']
            # update dropout
            target.model.update_dropout(new_h['dropout'])
            

def pbt_train(args):
    device = torch.device("cuda") if args.gpu else torch.device("mps") if args.mps else torch.device("cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.rundir, exist_ok=True)
    log_path = Path(args.rundir)/"population_log.jsonl"

    pop = []
    for i in range(args.population):
        init_h = {
            'learning_rate': np.random.uniform(1e-4,1e-2),
            'weight_decay' : np.random.uniform(1e-5,1e-3),
            'dropout'      : np.random.uniform(0.1,0.5),
            'aug_prob'     : np.random.uniform(0.0,1.0),
            # 'batch_size'   : Now always one because of memory allocation problems
        }
        pop.append(PBTMember(i, init_h, device, args.data_dir, args.labels_csv))

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        logs = []
        for m in pop:
            tl, ta, _, _ = run_model(m.model, m.train_loader, train=True,  optimizer=m.optimizer, eps=args.eps)
            vl, va, _, _ = run_model(m.model, m.val_loader,   train=False)
            m.scheduler.step(vl)
            if va > m.best_val_auc:
                m.best_val_auc = va
                m.checkpoint   = copy.deepcopy(m.model.state_dict())
            logs.append({
                'epoch': epoch,
                'model_id': m.id,
                'copied_from': m.copied_from,
                **m.hyperparams,
                'train_auc': ta,
                'val_auc':   va
            })
        # write JSONL
        with open(log_path,'a') as f:
            for entry in logs:
                f.write(json.dumps(entry)+'\n')
        # clear lineage
        for m in pop:
            m.copied_from = None
        # print best
        best = max(logs, key=lambda x:x['val_auc'])
        print(f"Best: id={best['model_id']} train_auc={best['train_auc']:.4f} val_auc={best['val_auc']:.4f}")
        # exploit
        if epoch % args.eval_interval == 0:
            exploit_and_explore(pop)

    # save final best
    best_m = max(pop, key=lambda m:m.best_val_auc)
    torch.save(best_m.checkpoint, Path(args.rundir)/'best_model.pth')
    with open(Path(args.rundir)/'best_hyperparams.json','w') as f:
        json.dump(best_m.hyperparams, f, indent=2)

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--rundir',      required=True)
    p.add_argument('--data_dir',    required=True)
    p.add_argument('--labels_csv',  required=True)
    p.add_argument('--seed',    type=int,   default=42)
    p.add_argument('--gpu',            action='store_true')
    p.add_argument('--mps',            action='store_true')
    p.add_argument('--epochs', type=int,   default=50)
    p.add_argument('--eps',    type=float, default=0.0)
    p.add_argument('--population', type=int, default=3)
    p.add_argument('--eval_interval', type=int, default=5)
    return p

if __name__=='__main__':
    args = get_parser().parse_args()
    pbt_train(args)
