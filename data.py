# data.py
import os, glob
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_txt(path: str) -> np.ndarray:
    # robust to commas/whitespace, no header
    return pd.read_csv(path, header=None).values.astype(np.float32)

def windowify(arr: np.ndarray, T: int, stride: int) -> np.ndarray:
    N, D = arr.shape
    if N < T: return np.zeros((0, T, D), dtype=arr.dtype)
    idxs = [(s, s+T) for s in range(0, N - T + 1, stride)]
    return np.stack([arr[s:e] for s, e in idxs], axis=0) if idxs else np.zeros((0, T, D), dtype=arr.dtype)

def window_labels(row_labels: np.ndarray, T: int, stride: int, policy: str = "any") -> np.ndarray:
    N = len(row_labels)
    idxs = [(s, s+T) for s in range(0, N - T + 1, stride)]
    out = []
    for s, e in idxs:
        win = row_labels[s:e]
        if policy == "any":
            out.append(1 if np.any(win > 0.5) else 0)
        else:
            out.append(1 if np.mean(win) > 0.5 else 0)
    return np.array(out, dtype=np.int64)

def load_train_windows(data_root: str, T: int, stride: int) -> np.ndarray:
    train_dir = os.path.join(data_root, "train")
    files = sorted(glob.glob(os.path.join(train_dir, "*.txt")))
    wins = []
    D = None
    for f in files:
        arr = load_txt(f)
        D = D or arr.shape[1]
        w = windowify(arr, T, stride)
        if len(w): wins.append(w)
    return np.concatenate(wins, axis=0) if wins else np.zeros((0, T, D or 1), np.float32)

def load_test_by_file(data_root: str, T: int, stride: int, policy: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    test_dir = os.path.join(data_root, "test")
    lbl_dir  = os.path.join(data_root, "test_labels")
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in sorted(glob.glob(os.path.join(test_dir, "*.txt"))):
        name = os.path.basename(f)
        arr  = load_txt(f)
        w    = windowify(arr, T, stride)
        lbl_path = os.path.join(lbl_dir, name)
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(f"Missing labels for {name} at {lbl_path}")
        lbl = load_txt(lbl_path).squeeze(-1)
        if lbl.ndim != 1 or len(lbl) != len(arr):
            raise ValueError(f"Label shape mismatch for {name}: labels {lbl.shape}, rows {arr.shape[0]}")
        wlbl = window_labels(lbl, T, stride, policy)
        if len(w) != len(wlbl):
            raise ValueError(f"Window count mismatch for {name}: {len(w)} vs {len(wlbl)}")
        out[name] = (w, wlbl)
    return out

def fit_scaler_on_windows(windows: np.ndarray) -> StandardScaler:
    N, T, D = windows.shape
    flat = windows.reshape(N*T, D)
    return StandardScaler().fit(flat)

def apply_scaler_windows(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if len(windows) == 0: return windows
    N, T, D = windows.shape
    flat = windows.reshape(N*T, D)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(N, T, D)

def concat_splits(by_file: Dict[str, Tuple[np.ndarray, np.ndarray]], names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    Ws, Ys = [], []
    for n in names:
        if n not in by_file:
            raise ValueError(f"{n} not found in test set.")
        w, y = by_file[n]
        if len(w):
            Ws.append(w); Ys.append(y)
    if not Ws:
        raise RuntimeError("Empty split after selection.")
    return np.concatenate(Ws, axis=0), np.concatenate(Ys, axis=0)
