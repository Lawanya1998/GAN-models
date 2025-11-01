# data.py
import os, glob
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LABEL_PREFIX = "Label"

def list_csvs(root: str, sub: str) -> List[str]:
    p1 = sorted(glob.glob(os.path.join(root, sub, "*.csv")))
    # also allow flat layout: root/*.csv (if subfolder missing)
    p2 = sorted(glob.glob(os.path.join(root, f"{sub}_*.csv")))
    return p1 if p1 else p2

def read_csv_file(path: str) -> pd.DataFrame:
    # Robust CSV reader; keep all columns as they are
    df = pd.read_csv(path)
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    return df

def detect_label_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("Label")]

def default_feature_columns(df: pd.DataFrame, label_name: str) -> List[str]:
    # Everything except Time + any label columns
    label_cols = detect_label_columns(df)
    feat = [c for c in df.columns if c not in (["Time"] + label_cols)]
    # drop the chosen label column if it accidentally sneaks in
    if label_name in feat:
        feat.remove(label_name)
    return feat

def windowify(arr: np.ndarray, T: int, stride: int) -> np.ndarray:
    N, D = arr.shape
    if N < T: return np.zeros((0, T, D), dtype=arr.dtype)
    idxs = [(s, s + T) for s in range(0, N - T + 1, stride)]
    if not idxs: return np.zeros((0, T, D), dtype=arr.dtype)
    return np.stack([arr[s:e] for s, e in idxs], axis=0)

def window_labels(row_labels: np.ndarray, T: int, stride: int, policy: str = "any") -> np.ndarray:
    # policy: "any" (if any row in the window is 1 -> window=1)
    #         "majority" (if >50% rows in the window are 1 -> window=1)
    N = len(row_labels)
    idxs = [(s, s + T) for s in range(0, N - T + 1, stride)]
    out = []
    for s, e in idxs:
        win = row_labels[s:e]
        if policy == "any":
            out.append(1 if np.any(win > 0.5) else 0)
        else:
            out.append(1 if np.mean(win > 0.5) > 0.5 else 0)
    return np.array(out, dtype=np.int64)

def load_split_windows(
    data_root: str,
    split: str,            # "train" or "test"
    T: int,
    stride: int,
    label_name: str,
    policy: str = "any"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns {filename: (windows [N,T,D], labels [N])} for the split.
    """
    files = list_csvs(data_root, split)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for f in files:
        name = os.path.basename(f)
        df = read_csv_file(f)
        if label_name not in df.columns:
            raise ValueError(f"Label column '{label_name}' not found in {name}. Available: {list(df.columns)}")

        # Forward/backward fill for any missing numeric values
        feat_cols = default_feature_columns(df, label_name=label_name)
        X = df[feat_cols].ffill().bfill().to_numpy(dtype=np.float32)
        y = df[label_name].astype(float).to_numpy()

        wX = windowify(X, T, stride)
        wy = window_labels(y, T, stride, policy)

        if len(wX) != len(wy):
            raise RuntimeError(f"Window count mismatch in {name}: {len(wX)} vs {len(wy)}")

        if len(wX):
            out[name] = (wX, wy)
    return out

def concat_splits(by_file: Dict[str, Tuple[np.ndarray, np.ndarray]], names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    Ws, Ys = [], []
    for n in names:
        if n not in by_file:
            raise ValueError(f"{n} not found in split.")
        w, y = by_file[n]
        if len(w):
            Ws.append(w); Ys.append(y)
    if not Ws:
        raise RuntimeError("Empty selection after filtering.")
    return np.concatenate(Ws, axis=0), np.concatenate(Ys, axis=0)

def fit_scaler_on_windows(windows: np.ndarray) -> StandardScaler:
    if len(windows) == 0:
        raise RuntimeError("Cannot fit scaler on empty windows.")
    N, T, D = windows.shape
    flat = windows.reshape(N*T, D)
    sc = StandardScaler()
    sc.fit(flat)
    return sc

def apply_scaler_windows(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if len(windows) == 0: return windows
    N, T, D = windows.shape
    flat = windows.reshape(N*T, D)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(N, T, D)

def split_train_XA(
    train_by_file: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X (normals, label==0) and A (anomalies, label==1) from TRAIN windows.
    """
    Xs, As = [], []
    for _, (w, y) in train_by_file.items():
        if len(w) == 0: continue
        Xs.append(w[y == 0])
        As.append(w[y == 1])
    X = np.concatenate([z for z in Xs if len(z)], axis=0) if any(len(z) for z in Xs) else np.zeros((0,1,1), np.float32)
    A = np.concatenate([z for z in As if len(z)], axis=0) if any(len(z) for z in As) else np.zeros((0,1,1), np.float32)
    return X, A
