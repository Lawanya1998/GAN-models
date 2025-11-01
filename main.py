# main.py
# main.py
import os, time, argparse
import numpy as np
import torch

from data import (
    load_split_windows, split_train_XA,
    fit_scaler_on_windows, apply_scaler_windows, concat_splits, list_csvs
)
from train import train_interp_gan

def parse_args():
    p = argparse.ArgumentParser(description="GAN-style interpolation anomaly detection (time-series windows)")
    p.add_argument("--data_root", type=str, default="data", help="folder containing train/ and test/")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--label_policy", type=str, default="any", choices=["any","majority"])
    p.add_argument("--label_name", type=str, default="Label (common/all)")
    p.add_argument("--rnn", type=str, default="gru", choices=["gru","lstm"])

    # Training
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam0", type=float, default=1.0)
    p.add_argument("--lam1", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=1337)

    # IO
    p.add_argument("--run_dir", type=str, default=None)
    return p.parse_args()

def set_seed(seed=1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.run_dir = os.path.join("runs", f"run-{ts}")
    os.makedirs(args.run_dir, exist_ok=True)

    # 1) Load TRAIN windows and split X(normal) / A(anomaly)
    train_by_file = load_split_windows(args.data_root, "train", args.window, args.stride, args.label_name, args.label_policy)
    X, A = split_train_XA(train_by_file)
    print(f"[TRAIN] normals={len(X)} anomalies={len(A)}  (window={args.window}, stride={args.stride})")
    if len(X)==0 or len(A)==0:
        raise RuntimeError("Need non-empty normals and anomalies in TRAIN.")

    # Fit scaler on TRAIN only (both X and A)
    train_all = np.concatenate([X, A], axis=0)
    scaler = fit_scaler_on_windows(train_all)
    X = apply_scaler_windows(X, scaler)
    A = apply_scaler_windows(A, scaler)

    # 2) Load TEST windows (auto: first file as VAL, rest as HOLDOUT)
    test_by_file = load_split_windows(args.data_root, "test", args.window, args.stride, args.label_name, args.label_policy)
    test_files = sorted(list(test_by_file.keys()))
    if not test_files:
        raise RuntimeError("No test CSVs found under data/test.")

    val_name = test_files[0]         # simple heuristic; change if you want
    hold_names = test_files[1:] if len(test_files) > 1 else []
    val_w, val_y = test_by_file[val_name]
    val_w = apply_scaler_windows(val_w, scaler)

    hold_splits = {}
    for n in hold_names:
        w, y = test_by_file[n]
        hold_splits[n] = (apply_scaler_windows(w, scaler), y)

    print(f"[VAL] {val_name}  windows={len(val_w)}")
    print(f"[HOLDOUT] files={len(hold_splits)}")

    # 3) Train + evaluate
    train_interp_gan(
        X, A,
        val_split=(val_w, val_y),
        hold_splits=hold_splits,
        hidden=args.hidden, epochs=args.epochs, batch=args.batch_size, lr=args.lr,
        lam0=args.lam0, lam1=args.lam1,
        rnn=args.rnn,
        device=device,
        run_dir=args.run_dir
    )

    print("\nDone. Artifacts in:", args.run_dir)
    print("  - best_D.pt, best_G.pt")
    print("  - val_stats.json, summary.json")

if __name__ == "__main__":
    main()
