# train.py
# train.py
import os, json
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix

from models import Discriminator, Generator

# ---------------------------
# Datasets
# ---------------------------
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray = None):
        self.W = torch.from_numpy(windows)  # [N,T,D]
        self.y = None if labels is None else torch.from_numpy(labels.astype(np.int64))

    def __len__(self): return len(self.W)

    def __getitem__(self, idx):
        if self.y is None:
            return self.W[idx]
        return self.W[idx], self.y[idx]

class PairInterpDataset(Dataset):
    """ Yields (x, a, gamma) where gamma ~ U(0.1, 0.9) """
    def __init__(self, X: np.ndarray, A: np.ndarray, g_lo=0.1, g_hi=0.9):
        assert len(X) and len(A), "X and A must be non-empty for interpolation training"
        self.X = torch.from_numpy(X)
        self.A = torch.from_numpy(A)
        self.g_lo, self.g_hi = g_lo, g_hi

    def __len__(self): return max(len(self.X), len(self.A))

    def __getitem__(self, _idx):
        i = torch.randint(0, len(self.X), (1,)).item()
        j = torch.randint(0, len(self.A), (1,)).item()
        x = self.X[i]
        a = self.A[j]
        g = torch.empty(1).uniform_(self.g_lo, self.g_hi).item()
        return x, a, torch.tensor(g, dtype=torch.float32)

# ---------------------------
# Metrics helpers
# ---------------------------
def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    out = {}
    # AUROC can be NaN if only one class in y_true
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["auroc"] = float("nan")
    out["aupr"] = float(average_precision_score(y_true, scores))
    # Best F2
    prec, rec, thr = precision_recall_curve(y_true, scores)
    beta2 = 2.0
    f2 = (1+beta2**2) * (prec*rec) / ((beta2**2)*prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f2))
    best_thr = thr[max(0, min(best_idx-1, len(thr)-1))] if len(thr) > 0 else 0.5
    out["best_f2"] = float(f2[best_idx]) if len(f2) else 0.0
    out["thr_star"] = float(best_thr)
    out["pos_rate"] = float((scores >= best_thr).mean())
    return out

def eval_one(D, windows: np.ndarray, labels: np.ndarray, device: str) -> Dict[str, float]:
    dl = DataLoader(WindowDataset(windows, labels), batch_size=64, shuffle=False)
    D.eval()
    scores, ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            s = D(xb.to(device)).detach().cpu().numpy()
            scores.append(s); ys.append(yb.numpy())
    scores = np.concatenate(scores, 0)
    ys     = np.concatenate(ys, 0)
    stats  = evaluate_scores(ys, scores)
    thr    = stats["thr_star"]
    yhat   = (scores >= thr).astype(int)
    stats["f1"] = float(f1_score(ys, yhat, zero_division=0))
    cm = confusion_matrix(ys, yhat, labels=[0,1])
    stats["cm"] = cm.tolist()
    return stats

# ---------------------------
# Training loop (GAN-style interpolation training)
# ---------------------------
def train_interp_gan(
    X: np.ndarray, A: np.ndarray,
    val_split: Tuple[np.ndarray, np.ndarray],
    hold_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    hidden: int = 64, epochs: int = 30, batch: int = 64, lr: float = 1e-3,
    lam0: float = 1.0, lam1: float = 1.5,
    rnn: str = "gru",
    device: str = "cpu",
    run_dir: str = "runs/run"
):
    os.makedirs(run_dir, exist_ok=True)
    D = Discriminator(input_dim=X.shape[2], hidden=hidden, rnn=rnn).to(device)
    G = Generator(input_dim=X.shape[2], hidden=hidden, rnn=rnn).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=lr)
    optG = torch.optim.Adam(G.parameters(), lr=lr)

    train_dl = DataLoader(PairInterpDataset(X, A), batch_size=batch, shuffle=True, drop_last=False)
    val_w, val_y = val_split

    # Train
    best_val_aupr = -1.0
    for ep in range(1, epochs+1):
        D.train(); G.train()
        pbar = tqdm(train_dl, desc=f"[Interp] epoch {ep}/{epochs}")
        for x, a, gamma in pbar:
            x = x.to(device); a = a.to(device); gamma = gamma.to(device)

            # ---- D step: gamma regression on generated y = G(x,a,g) ----
            with torch.no_grad():
                y = G(x, a, gamma)
            pred_g = D(y)
            lossD = nn.functional.mse_loss(pred_g, gamma)

            optD.zero_grad()
            lossD.backward()
            optD.step()

            # ---- G step: match anchors + keep D(y_mid) close to gamma ----
            y_mid = G(x, a, gamma)
            pred_mid = D(y_mid)
            L_interp = nn.functional.mse_loss(pred_mid, gamma)

            y0 = G(x, a, torch.zeros_like(gamma))
            y1 = G(x, a, torch.ones_like(gamma))
            L0 = nn.functional.mse_loss(y0, x)
            L1 = nn.functional.mse_loss(y1, a)

            lossG = L_interp + lam0 * L0 + lam1 * L1

            optG.zero_grad()
            lossG.backward()
            optG.step()

            pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()), L0=float(L0.item()), L1=float(L1.item()))

        # ---- Validation (score real windows with D) ----
        val_stats = eval_one(D, val_w, val_y, device)
        print(f"[VAL] AUROC={val_stats['auroc']:.3f}  AUPR={val_stats['aupr']:.3f}  best-F2={val_stats['best_f2']:.3f}  thr*={val_stats['thr_star']:.3f}")
        if val_stats["aupr"] > best_val_aupr + 1e-6:
            best_val_aupr = val_stats["aupr"]
            torch.save(D.state_dict(), os.path.join(run_dir, "best_D.pt"))
            torch.save(G.state_dict(), os.path.join(run_dir, "best_G.pt"))
            with open(os.path.join(run_dir, "val_stats.json"), "w") as f:
                json.dump(val_stats, f, indent=2)

    # Final evaluation (best D)
    D.load_state_dict(torch.load(os.path.join(run_dir, "best_D.pt"), map_location=device))
    D.eval()
    all_cm = np.array([[0,0],[0,0]], dtype=int)

    # Recompute on validation
    val_stats = eval_one(D, val_w, val_y, device)
    print("\n[FINAL VAL]")
    print(json.dumps(val_stats, indent=2))

    # Holdout per file + aggregate
    summary = {"val": val_stats, "holdout": {}}
    for name, (w, y) in hold_splits.items():
        s = eval_one(D, w, y, device)
        print(f"\n[HOLDOUT] {name}")
        print(json.dumps(s, indent=2))
        all_cm += np.array(s["cm"])
        summary["holdout"][name] = s

    summary["holdout_all_cm"] = all_cm.tolist()
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[HOLDOUT][ALL] Confusion Matrix (labels=[0,1]):")
    print(all_cm)
