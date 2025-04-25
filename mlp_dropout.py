#!/usr/bin/env python
"""
Example running commands:
python mlp_dropout.py \
  --data heart_failure_clinical_records_dataset.csv \
  --label_name DEATH_EVENT \
  --test_size 0.1 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.1 \
  --hidden_dims 16 16 16 \
  --dropout 0.4 \
  --device mps \
  --seed 45

"""
import argparse
import csv
import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            dropout: float,
            dropout_infer: float,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.p_train = dropout
        self.p_infer = dropout_infer
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        rate = self.p_train if self.training else self.p_infer
        x = F.dropout(x, p=rate, training=True)
        x = self.linear(x)
        x = self.activation(x)
        y = self.dropout(x)
        return y


class ResMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] | None = None,
            num_classes: int = 2,
            dropout: float = 0.1,
            dropout_infer: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [16, 16]
        layers = [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(MLP(in_dim, out_dim, dropout=dropout, dropout_infer=dropout_infer))
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    # 1. Python built‐in random
    random.seed(seed)
    # 2. NumPy
    np.random.seed(seed)
    # 3. Torch (CPU)
    torch.manual_seed(seed)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tx, ty = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optim, device) -> float:
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def set_inference_dropout(model: nn.Module, new_dropout_rate: float):
    """
    Walks the model and for every nn.Dropout:
     • sets its drop‐probability to p
     • turns it into train‐mode so it actually drops
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = new_dropout_rate
            m.train()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, repeats=3) -> (List, float):
    repeated_logits, preds, gts = [], [], []
    for _ in range(repeats):
        logits = []
        for xb, yb in loader:
            logit = model(xb.to(device)).cpu()
            logits.append(logit)
            predicts = logit.argmax(dim=-1)
            preds.append(predicts)
            gts.append(yb)
        logits = torch.cat(logits)
        repeated_logits.append(logits)
    repeated_logits = torch.stack(repeated_logits)
    logits = torch.softmax(repeated_logits, dim=-1)[..., 1]

    # logits shape: (repeats, test_size)
    return logits, accuracy_score(torch.cat(gts), torch.cat(preds))


def main():
    parser = argparse.ArgumentParser("MLP classifier")
    parser.add_argument("--data", required=True)
    parser.add_argument("--label_name", default="label")
    parser.add_argument("--sex_name", default="sex")
    parser.add_argument("--age_name", default="age")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_dropout_original", type=float, default=0.2)
    parser.add_argument("--eval_dropout_masked", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file_path", type=str, default="prob_output.csv")
    parser.add_argument("--repeats", type=int, default=10, help="Number of times to run the model")

    args = parser.parse_args()
    set_seed(args.seed)

    # 1. Load data
    rng = np.random.RandomState(args.seed)
    df = pd.read_csv(args.data)

    # 2.The two columns of interest
    sex_idx = df.columns.get_loc(args.sex_name)
    age_idx = df.columns.get_loc(args.age_name)

    y = df.pop(args.label_name).to_numpy()
    X = df.to_numpy(dtype=np.float32)

    # 3. Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    counts = np.bincount(y_tr)
    class_w = counts.sum() / (len(counts) * counts)
    class_w = torch.as_tensor(class_w, dtype=torch.float32, device=args.device)

    # 4. Create permuted-column copy of the test features
    # TODO(yixiao): can try different masking methods.
    X_te_perm = X_te.copy()
    for col in (sex_idx, age_idx):
        X_te_perm[:, col] = rng.permutation(X_te_perm[:, col])

    # 5. DataLoaders
    tr_loader = make_loader(X_tr, y_tr, args.batch_size, True)
    te_loader = make_loader(X_te, y_te, args.batch_size, False)
    perm_loader = make_loader(X_te_perm, y_te, args.batch_size, False)

    # 6. Model / loss / optimizer / scheduler
    model = ResMLP(X.shape[1], args.hidden_dims, len(np.unique(y)), args.dropout, args.eval_dropout_original).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    t_max = args.epochs  # default period = total epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, eta_min=1e-5, T_max=t_max)

    # 7. Training loop
    best_acc = 0.0
    best_ckpt = "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr_loader, criterion, optim, args.device)
        _, acc_orig = evaluate(model, te_loader, args.device)
        # if this epoch is the new best, save a checkpoint
        if acc_orig > best_acc:
            best_acc = acc_orig
            torch.save(model.state_dict(), best_ckpt)
        scheduler.step()
        lr_now = optim.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | lr: {lr_now:.2e} | loss: {tr_loss:.4f} "
              f"| test acc: {acc_orig:.4f} | best: {best_acc:.4f}")

    # 8. Final evaluation on both test variants
    model1 = ResMLP(X.shape[1], args.hidden_dims, len(np.unique(y)), args.dropout, args.eval_dropout_original).to(args.device)
    model2 = ResMLP(X.shape[1], args.hidden_dims, len(np.unique(y)), args.dropout, args.eval_dropout_masked).to(args.device)

    model1.load_state_dict(torch.load(best_ckpt, map_location=args.device))
    model1.to(args.device)
    model1.eval()
    model2.load_state_dict(torch.load(best_ckpt, map_location=args.device))
    model2.to(args.device)
    model2.eval()

    logits1, acc_orig = evaluate(model1, te_loader, args.device, repeats=args.repeats)
    logits2, acc_perm = evaluate(model2, perm_loader, args.device, repeats=args.repeats)
    print(f"\nFinal accuracy — original test set:  {acc_orig:.4f}")
    print(f"Final accuracy — permuted sex/age set: {acc_perm:.4f}")

    num_repeats, num_records = logits1.shape
    record_ids = torch.arange(1, num_records + 1).repeat_interleave(num_repeats)

    score1_flat = logits1.T.contiguous().view(-1)
    score2_flat = logits2.T.contiguous().view(-1)

    df = pd.DataFrame({
        'record_id': record_ids.numpy(),
        'prob_original': score1_flat.numpy(),
        'prob_masked': score2_flat.numpy()
    })
    df.to_csv('prob_output.csv', index=False)


if __name__ == "__main__":
    main()
