# train.py
import os, math, random, time
from typing import Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

from models import LSTMAE, Discriminator, Generator

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Datasets
# ---------------------------
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None):
        self.W = torch.from_numpy(windows)  # [N,T,D]
        self.y = None if labels is None else torch.from_numpy(labels.astype(np.int64))
    def __len__(self): return len(self.W)
    def __getitem__(self, idx):
        return (self.W[idx], self.y[idx]) if self.y is not None else self.W[idx]

class PairInterpDataset(Dataset):
    # yields (x, a, gamma) with gamma ~ U(0.1,0.9)
    def __init__(self, X: np.ndarray, A: np.ndarray, g_lo=0.1, g_hi=0.9):
        self.X = torch.from_numpy(X)
        self.A = torch.from_numpy(A)
        self.g_lo = g_lo; self.g_hi = g_hi
    def __len__(self): return max(len(self.X), len(self.A))
    def __getitem__(self, idx):
        i = torch.randint(0, len(self.X), (1,)).item()
        j = torch.randint(0, len(self.A), (1,)).item()
        x = self.X[i]; a = self.A[j]
        g = torch.empty(1).uniform_(self.g_lo, self.g_hi).item()
        return x, a, torch.tensor(g, dtype=torch.float32)

# ---------------------------
# AE bootstrap split
# ---------------------------
def bootstrap_train_split(train_windows: np.ndarray, hidden=64, epochs=15, batch=128, lr=1e-3,
                          device="cpu", p_norm=0.7, p_anom=0.2):
    N, T, D = train_windows.shape
    model = LSTMAE(input_dim=D, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = WindowDataset(train_windows, labels=None)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    model.train()
    for ep in range(epochs):
        tot = 0.0; cnt = 0
        for xb in dl:
            xb = xb.to(device)
            yb = model(xb)
            loss = nn.functional.mse_loss(yb, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); cnt += len(xb)
        print(f"[AE] epoch {ep+1}/{epochs} recon_loss={tot/max(1,cnt):.6f}")

    # errors
    model.eval()
    errs = []
    with torch.no_grad():
        for xb in DataLoader(ds, batch_size=batch, shuffle=False):
            xb = xb.to(device)
            yb = model(xb)
            e = nn.functional.mse_loss(yb, xb, reduction="none").mean(dim=(1,2))
            errs.append(e.cpu().numpy())
    errs = np.concatenate(errs, axis=0)

    lo_thr = np.quantile(errs, p_norm)        # normals: lowest errors
    hi_thr = np.quantile(errs, 1.0 - p_anom)  # anomalies: highest errors
    X = train_windows[errs <= lo_thr]
    A = train_windows[errs >= hi_thr]
    print(f"[AE] Split -> X (normals): {len(X)} | A (anomalies): {len(A)} | total: {len(train_windows)}")
    return X, A

# ---------------------------
# Eval helpers
# ---------------------------
def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, scores)
    aupr  = average_precision_score(y_true, scores)
    # Best F2 threshold on validation
    prec, rec, thr = precision_recall_curve(y_true, scores)
    beta2 = 2.0
    f2 = (1+beta2**2) * (prec*rec) / ((beta2**2)*prec + rec + 1e-12)
    best_idx = np.nanargmax(f2)
    best_thr = thr[max(0, best_idx-1)] if best_idx-1 >= 0 and best_idx-1 < len(thr) else 0.5
    best_f2  = f2[best_idx]
    pos_rate = float((scores >= best_thr).mean())
    return {"auroc": auroc, "aupr": aupr, "best_f2": float(best_f2), "thr_star": float(best_thr), "pos_rate": pos_rate}

def binarize(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores >= thr).astype(int)

def classification_report(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    f1  = f1_score(y_true, y_hat, zero_division=0)
    acc = float((y_true == y_hat).mean())
    # Precision@k%: top-k scores marked positive
    out = {"f1": float(f1), "acc": acc}
    return out

def precision_at_k(scores: np.ndarray, y_true: np.ndarray, k: float) -> float:
    n = len(scores); k_n = max(1, int(math.ceil(k * n)))
    idx = np.argsort(scores)[::-1][:k_n]
    return float(y_true[idx].mean())

# ---------------------------
# Interpolation training loop
# ---------------------------
def train_interp(X: np.ndarray, A: np.ndarray,
                 val_w: np.ndarray, val_y: np.ndarray,
                 hold_w: np.ndarray, hold_y: np.ndarray,
                 hidden=64, epochs=50, batch=32, lr=1e-3,
                 lam0=1.0, lam1=1.5, device="cpu",
                 patience=6, run_dir="runs"):
    os.makedirs(run_dir, exist_ok=True)
    D = Discriminator(input_dim=X.shape[2], hidden=hidden).to(device)
    G = Generator(input_dim=X.shape[2], hidden=hidden).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=lr)
    optG = torch.optim.Adam(G.parameters(), lr=lr)

    train_dl = DataLoader(PairInterpDataset(X, A, 0.1, 0.9), batch_size=batch, shuffle=True, drop_last=False)
    val_dl   = DataLoader(WindowDataset(val_w, labels=val_y), batch_size=batch, shuffle=False)
    hold_dl  = DataLoader(WindowDataset(hold_w, labels=hold_y), batch_size=batch, shuffle=False)

    best_aupr = -1.0
    no_improve = 0

    def get_scores(dataloader):
        D.eval()
        scores, ys = [], []
        with torch.no_grad():
            for xb, yb in dataloader:
                s = D(xb.to(device))
                scores.append(s.cpu().numpy()); ys.append(yb.numpy())
        return np.concatenate(scores,0), np.concatenate(ys,0)

    for ep in range(epochs):
        D.train(); G.train()
        pbar = tqdm(train_dl, desc=f"[Interp] epoch {ep+1}/{epochs}")
        for x, a, gamma in pbar:
            x = x.to(device); a = a.to(device); gamma = gamma.to(device)

            # 1) D step (regress gamma on generated)
            with torch.no_grad():
                y = G(x, a, gamma)
            pred = D(y)
            lossD = nn.functional.mse_loss(pred, gamma)
            optD.zero_grad(); lossD.backward(); optD.step()

            # 2) G step (interpolation + anchors)
            y_mid = G(x, a, gamma)
            pred_mid = D(y_mid)
            L_interp = nn.functional.mse_loss(pred_mid, gamma)
            y0 = G(x, a, torch.zeros_like(gamma))
            y1 = G(x, a, torch.ones_like(gamma))
            L0 = nn.functional.mse_loss(y0, x)
            L1 = nn.functional.mse_loss(y1, a)
            lossG = L_interp + lam0 * L0 + lam1 * L1
            optG.zero_grad(); lossG.backward(); optG.stop_gradient = False; optG.step()
            pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()), L0=float(L0.item()), L1=float(L1.item()))

        # ---- VAL ----
        val_scores, val_true = get_scores(val_dl)
        val_stats = evaluate_scores(val_true, val_scores)
        print(f"[Eval-VAL] AUROC={val_stats['auroc']:.4f}  AUPR={val_stats['aupr']:.4f}  Best-F2={val_stats['best_f2']:.4f} thr*={val_stats['thr_star']:.3f}")

        # early stopping on AUPR
        if val_stats["aupr"] > best_aupr + 1e-6:
            best_aupr = val_stats["aupr"]
            no_improve = 0
            torch.save(D.state_dict(), os.path.join(run_dir, "best_D.pt"))
            torch.save(G.state_dict(), os.path.join(run_dir, "best_G.pt"))
            np.save(os.path.join(run_dir, "val_thr_star.npy"), np.array([val_stats["thr_star"]], dtype=np.float32))
            np.save(os.path.join(run_dir, "val_pos_rate.npy"), np.array([val_stats["pos_rate"]], dtype=np.float32))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep+1} (no AUPR improvement for {patience} epochs).")
                break

    # ---- Final evaluation with best D ----
    D.load_state_dict(torch.load(os.path.join(run_dir, "best_D.pt"), map_location=device))
    D.eval()

    # Recompute VAL/HOLDOUT metrics
    val_scores, val_true = [], []
    hold_scores, hold_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            val_scores.append(D(xb.to(device)).cpu().numpy()); val_true.append(yb.numpy())
        for xb, yb in hold_dl:
            hold_scores.append(D(xb.to(device)).cpu().numpy()); hold_true.append(yb.numpy())
    val_scores = np.concatenate(val_scores,0); val_true = np.concatenate(val_true,0)
    hold_scores = np.concatenate(hold_scores,0); hold_true = np.concatenate(hold_true,0)

    val_stats = evaluate_scores(val_true, val_scores)
    print("\n=== VALIDATION (for tuning) ===")
    print(f"  AUROC: {val_stats['auroc']:.3f}  AUPR: {val_stats['aupr']:.3f}  Best-F2: {val_stats['best_f2']:.3f}  thr*: {val_stats['thr_star']:.3f}")
    print(f"  PosRate@thr*: {val_stats['pos_rate']:.3f}")

    thr_fixed = float(val_stats["thr_star"])
    pos_rate  = float(val_stats["pos_rate"])
    thr_rate  = float(np.quantile(hold_scores, 1.0 - pos_rate + 1e-9))

    # HOLDOUT with both thresholds
    print("\n=== HOLDOUT (final) ===")
    print(f"  AUROC: {roc_auc_score(hold_true, hold_scores):.3f}  AUPR: {average_precision_score(hold_true, hold_scores)::.3f}")

    # fixed threshold
    hold_pred_fixed = binarize(hold_scores, thr_fixed)
    rpt_fixed = classification_report(hold_true, hold_pred_fixed)
    p5 = precision_at_k(hold_scores, hold_true, 0.05)
    p10= precision_at_k(hold_scores, hold_true, 0.10)
    print("  -- Using fixed VAL threshold ({:.3f}) --".format(thr_fixed))
    print("     ACC: {:.3f}  F1: {:.3f}  P@5%: {:.3f}  P@10%: {:.3f}".format(rpt_fixed["acc"], rpt_fixed["f1"], p5, p10))

    # rate-matched
    hold_pred_rate = binarize(hold_scores, thr_rate)
    rpt_rate = classification_report(hold_true, hold_pred_rate)
    p5r = precision_at_k(hold_scores, hold_true, 0.05)
    p10r= precision_at_k(hold_scores, hold_true, 0.10)
    print("  -- Using rate-matched threshold ({:.3f}) --".format(thr_rate))
    print("     ACC: {:.3f}  F1: {:.3f}  P@5%: {:.3f}  P@10%: {:.3f}".format(rpt_rate["acc"], rpt_rate["f1"], p5r, p10r))

    return {
        "val": val_stats,
        "hold_fixed": rpt_fixed,
        "hold_rate": rpt_rate,
        "thr_fixed": thr_fixed,
        "thr_rate": thr_rate
    }
