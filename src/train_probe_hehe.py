# python train_probe_hehe.py --train /disk/u/harshraj/CotIF/data/Bespoke-Stratos-17k-llama3-hehe-partial_soln-meta-3.2.jsonl --val /disk/u/harshraj/CotIF/data/Bespoke-Stratos-17k-llama3-hehe-partial_soln-meta-test-3.2.jsonl --outdir probe_out-partial_soln  --batch_size 4

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Iterator, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig


class Probe(nn.Module):
    """Two‑layer MLP probe (linear when *hidden_dim* = 0)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1, bias=True),
            )
        else: 
            self.net = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.sigmoid(self.net(x).flatten())

def expected_calibration_error(probs: torch.Tensor,
                               labels: torch.Tensor,
                               n_bins: int = 15) -> float:
    """Guo et al. (2017) ECE over equal-width bins on [0, 1]."""
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        in_bin = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if in_bin.any():
            bin_conf = probs[in_bin].mean()
            bin_acc  = labels[in_bin].float().mean()
            ece += in_bin.float().mean() * (bin_conf - bin_acc).abs()
    return ece.item()

def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean-squared error between probabilities and binary labels."""
    return torch.square(probs - labels.float()).mean().item()

def weighted_bce(pred: torch.Tensor, tgt: torch.Tensor, neg_pos_ratio: float, alpha: float) -> torch.Tensor:
    weight = torch.where(tgt == 1, torch.tensor(neg_pos_ratio * alpha, device=tgt.device), torch.tensor(1.0, device=tgt.device))
    return nn.functional.binary_cross_entropy(pred, tgt.float(), weight=weight, reduction="mean")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def jsonl_reader(path: str) -> Iterator[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_features(
    jsonl_path: str,
    model_name: str,
    batch_size: int,
    device: str,
    max_length: int = 2048,
) -> List[Tuple[np.ndarray, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<partial_solution>", "</partial_solution>"]})
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    ps_token_id = tokenizer.convert_tokens_to_ids("<partial_solution>")

    data=[]
    texts=[]
    lbls=[]

    def flush_batch():
        nonlocal texts, lbls
        if not texts:
            return

        with torch.no_grad():
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            hidden_states = model(**enc).hidden_states[-1]

            input_ids = enc["input_ids"]  
            mask = input_ids == ps_token_id
            # if not mask.any():
            #     raise ValueError("No <partial_solution> token found in batch")
            positions = mask.float().argmax(dim=1).long() 
            reps = hidden_states[torch.arange(hidden_states.size(0)), positions]

        reps_np = reps.cpu().numpy()
        for rep_vec, lbl in zip(reps_np, lbls):
            data.append((rep_vec, lbl))

        texts, lbls = [], []

    _data = jsonl_reader(jsonl_path)
    for obj in tqdm(_data, desc=f"Extract {os.path.basename(jsonl_path)}"):
        for soln, chunk in zip(obj["partial_solutions-annotation"], obj["partial_solutions-chunks"]):
            corr = soln["result"]
            texts.append(chunk)
            lbls.append(int(bool(corr)))

            if len(texts) >= batch_size:
                flush_batch()
    flush_batch()
    return [a for a, _ in data], [b for _, b in data]

def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, neg_pos: float, alpha: float, device: str) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = weighted_bce(pred, y, neg_pos, alpha)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, float, float]:
    model.eval()
    preds, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        preds.append(model(x).cpu())
        labs.append(y)
    preds = torch.cat(preds)
    labs = torch.cat(labs)    
    acc   = (preds.round() == labs).float().mean().item()
    loss  = nn.functional.binary_cross_entropy(preds, labs.float()).item()
    ece   = expected_calibration_error(preds, labs)
    brier = brier_score(preds, labs)
    return acc, loss, ece, brier

def grid_search(
    train: Tuple[np.ndarray, np.ndarray],
    val: Tuple[np.ndarray, np.ndarray],
    grid: Dict[str, List],
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
) -> Tuple[nn.Module, Dict[str, float], float]:
    X_tr, y_tr = train
    X_val, y_val = val
    input_dim = X_tr.shape[1]
    neg_pos_ratio = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), shuffle=True, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, pin_memory=True)

    best_model, best_cfg, best_acc = None, None, -1.0

    for lr in grid["lr"]:
        for hidden in grid["hidden"]:
            for alpha in grid["alpha"]:
                for wd in grid["wd"]:
                    model = Probe(input_dim, hidden).to(device)
                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                    best_loss = float("inf")
                    best_state = None
                    epochs_no_improve = 0

                    for epoch in range(max_epochs):
                        _ = train_epoch(model, train_loader, opt, neg_pos_ratio, alpha, device)
                        acc, val_loss, _, _ = evaluate(model, val_loader, device)

                        if val_loss < best_loss - 1e-4:
                            best_loss = val_loss
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            break

                    model.load_state_dict(best_state)
                    acc, _, ece, brier = evaluate(model, val_loader, device)
                    if acc > best_acc or (acc == best_acc and (best_cfg is None or hidden < best_cfg["hidden"])):
                        best_model, best_cfg, best_acc = model, {"lr": lr, "hidden": hidden, "alpha": alpha, "wd": wd}, acc

    return best_model, best_cfg, best_acc


def load_dataset(path: str, model_name: Optional[str], batch_size: int, device: str, outdir: str) -> Tuple[np.ndarray, np.ndarray]:
    if path.endswith(".npz"):
        npz = np.load(path)
        return npz["features"], npz["labels"]
    elif path.endswith(".jsonl"):
        feats, lbls = extract_features(path, model_name, batch_size=batch_size, device=device)
        npz_path = f"{outdir}/extracted_features/{path.split('/')[-1].replace('.jsonl', '.npz')}"
        os.makedirs(f"{outdir}/extracted_features", exist_ok=True)
        print(f"💾 Saving features to {npz_path}")
        np.savez(npz_path, features=feats, labels=lbls)
        return load_dataset(path=npz_path, model_name=model_name, batch_size=batch_size, device=device, outdir=outdir)
    else:
        raise ValueError("Unsupported file format: {}".format(path))


def main():
    parser = argparse.ArgumentParser(description="Train a two‑layer MLP probe for answer‑correctness classification.")
    parser.add_argument("--train", required=True, help="Path to training .npz or .jsonl file.")
    parser.add_argument("--val", required=True, help="Path to validation .npz or .jsonl file.")
    parser.add_argument("--outdir", default="probe_out", help="Directory to save the trained probe.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HF model name/path to compute representations when using .jsonl input.")
    parser.add_argument("--grid", type=str, default='{"lr":[1e-4,3e-4,1e-3],"hidden":[0,32,128],"alpha":[1.0,2.0,5.0],"wd":[0.0,1e-4]}', help="JSON string specifying hyper‑parameter grid.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true", help="If set, run in evaluation-only mode.")
    parser.add_argument("--probe_model_path", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda | cpu")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    X_train, y_train = load_dataset(args.train, args.model, args.batch_size, args.device, args.outdir)
    X_val, y_val     = load_dataset(args.val,   args.model, args.batch_size, args.device, args.outdir)

    if not args.eval_only:
        print("Training samples:", X_train.shape[0], " | Validation samples:", X_val.shape[0])

        grid = json.loads(args.grid)

        model, cfg, acc = grid_search(
            (X_train, y_train),
            (X_val, y_val),
            grid,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
        )

        print("Best config:", cfg)
        print(f"Validation accuracy: {acc:.4f}")

        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "config": cfg,
        }, os.path.join(args.outdir, "probe.pt"))

        print(f"Probe saved to {os.path.join(args.outdir, 'probe.pt')}")
    
    if args.probe_model_path:
        ckpt = torch.load(args.probe_model_path, map_location=args.device)
    else:
        ckpt = torch.load(os.path.join(args.outdir, 'probe.pt'), map_location=args.device)
    input_dim = ckpt["input_dim"]
    hidden    = ckpt["config"]["hidden"]

    # 2. Reconstruct model and load weights
    model = Probe(input_dim=input_dim, hidden_dim=hidden).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Build loaders for final reporting
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).long()),
        batch_size=args.batch_size,
        pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).long()),
        batch_size=args.batch_size,
        pin_memory=True)

    # In-distribution (train) metrics
    tr_acc, tr_loss, tr_ece, tr_brier = evaluate(model, train_loader, args.device)
    # Out-distribution (val) metrics
    vd_acc, vd_loss, vd_ece, vd_brier = evaluate(model, val_loader,   args.device)

    print("─── Final probe performance ───")
    print(f"In-dist  ▶ Acc: {tr_acc:.4f} | Loss: {tr_loss:.4f} | "
          f"ECE: {tr_ece:.5f} | Brier: {tr_brier:.5f}")
    print(f"Out-dist ▶ Acc: {vd_acc:.4f} | Loss: {vd_loss:.4f} | "
          f"ECE: {vd_ece:.5f} | Brier: {vd_brier:.5f}")


if __name__ == "__main__":
    main()
