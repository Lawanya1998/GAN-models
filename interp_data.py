# interp_data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd  # only used in optional counting

from data import NoBoomDataset

def windowify(arr: np.ndarray, T: int, stride: int) -> np.ndarray:
    N, D = arr.shape
    if N < T: return np.zeros((0, T, D), dtype=arr.dtype)
    idxs = [(s, s+T) for s in range(0, N - T + 1, stride)]
    return np.stack([arr[s:e] for s, e in idxs], axis=0) if idxs else np.zeros((0, T, D), dtype=arr.dtype)

def window_labels(row_labels: np.ndarray, T: int, stride: int, policy="majority") -> np.ndarray:
    N = len(row_labels)
    idxs = [(s, s+T) for s in range(0, N - T + 1, stride)]
    out = []
    for s, e in idxs:
        win = row_labels[s:e]
        if policy == "any":
            out.append(1 if np.any(win > 0.5) else 0)
        else:
            out.append(1 if np.mean(win > 0.5) >= 0.5 else 0)
    return np.array(out, dtype=np.int64)

def build_X_A_from_train(train_ds: NoBoomDataset, T: int, stride: int):
    Xs, As = [], []
    for i in range(len(train_ds)):
        feats = train_ds.data[i]                       # [N,D]
        labels = train_ds.targets[i].astype(np.float32)# [N]
        w = windowify(feats, T, stride)                # [M,T,D]
        wy = window_labels(labels, T, stride, "majority")
        if len(w) == 0: continue
        if np.any(wy == 0): Xs.append(w[wy == 0])
        if np.any(wy == 1): As.append(w[wy == 1])
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, T, feats.shape[1]), np.float32)
    A = np.concatenate(As, axis=0) if As else np.zeros((0, T, feats.shape[1]), np.float32)
    return X, A

class PairInterpDataset(Dataset):
    """Yields (x, a, gamma) with gamma~U(0,1)."""
    def __init__(self, X: np.ndarray, A: np.ndarray):
        assert X.ndim == 3 and A.ndim == 3 and X.shape[1:] == A.shape[1:]
        self.X = torch.from_numpy(X)  # [Nx,T,D]
        self.A = torch.from_numpy(A)  # [Na,T,D]

    def __len__(self): return max(len(self.X), len(self.A))

    def __getitem__(self, idx):
        ix = torch.randint(0, len(self.X), (1,)).item()
        ia = torch.randint(0, len(self.A), (1,)).item()
        x = self.X[ix]  # [T,D]
        a = self.A[ia]  # [T,D]
        gamma = torch.rand(1).item()  # (0,1)
        return x, a, torch.tensor(gamma, dtype=torch.float32)

def score_test_windows(ds: NoBoomDataset, D_model, T: int, stride: int, device) -> tuple[np.ndarray, np.ndarray]:
    D_model.eval()
    all_wins, all_scores = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            arr = ds.data[i]                    # [N,D]
            w = windowify(arr, T, stride)       # [M,T,D]
            if len(w) == 0: continue
            x = torch.from_numpy(w).to(device).float()
            s = D_model(x).cpu().numpy()        # [M] in [0,1]
            all_wins.append(w); all_scores.append(s)
    if not all_wins:
        return np.zeros((0, T, ds.num_features), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(all_wins), np.concatenate(all_scores)
