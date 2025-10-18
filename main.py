# main.py
import os, argparse, time
import numpy as np
import torch

from data import (
    load_train_windows, load_test_by_file,
    fit_scaler_on_windows, apply_scaler_windows, concat_splits
)
from train import set_seed, bootstrap_train_split, train_interp

def parse_args():
    p = argparse.ArgumentParser(description="Interpolation-based anomaly scoring")
    p.add_argument("--data_root", type=str, default="data", help="path containing train/, test/, test_labels/")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--label_policy", type=str, default="any", choices=["any", "majority"])

    # Model/opt
    p.add_argument("--ae_hidden", type=int, default=64)
    p.add_argument("--ae_epochs", type=int, default=15)
    p.add_argument("--interp_hidden", type=int, default=64)
    p.add_argument("--interp_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam0", type=float, default=1.0)
    p.add_argument("--lam1", type=float, default=1.5)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=1337)

    # Bootstrap split
    p.add_argument("--p_norm", type=float, default=0.8, help="fraction kept as normals (<= percentile)")
    p.add_argument("--p_anom", type=float, default=0.1, help="fraction kept as anomalies (>= percentile from top)")

    # VAL / HOLDOUT filenames
    default_val = "machine-1-3.txt"
    default_hold = "machine-1-7.txt"
    p.add_argument("--val_files", type=str, default=default_val, help="comma-separated test file names for validation")
    p.add_argument("--holdout_files", type=str, default=default_hold, help="comma-separated test file names for holdout")

    # IO
    p.add_argument("--run_dir", type=str, default=None, help="output dir (default runs/run-<ts>)")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output dir
    if args.run_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.run_dir = os.path.join("runs", f"run-{ts}")
    os.makedirs(args.run_dir, exist_ok=True)

    # 1) Load data
    train_w = load_train_windows(args.data_root, args.window, args.stride)
    by_file = load_test_by_file(args.data_root, args.window, args.stride, args.label_policy)
    print("Train windows:", train_w.shape, "| Test files:", list(by_file.keys()))

    # 2) Scale (fit on train only)
    scaler = fit_scaler_on_windows(train_w)
    train_w = apply_scaler_windows(train_w, scaler)

    # Compose VAL/HOLDOUT splits by filename
    val_names = [s.strip() for s in args.val_files.split(",") if s.strip()]
    hold_names= [s.strip() for s in args.holdout_files.split(",") if s.strip()]

    val_w, val_y     = concat_splits(by_file, val_names)
    hold_w, hold_y   = concat_splits(by_file, hold_names)

    # scale test splits
    val_w  = apply_scaler_windows(val_w, scaler)
    hold_w = apply_scaler_windows(hold_w, scaler)

    print("VAL windows:", val_w.shape, "HOLDOUT windows:", hold_w.shape)

    if len(train_w)==0 or len(val_w)==0 or len(hold_w)==0:
        raise RuntimeError("Empty windows in one of the splits.")

    # 3) Bootstrap split (AE)
    X, A = bootstrap_train_split(train_w,
                                 hidden=args.ae_hidden, epochs=args.ae_epochs,
                                 batch=max(32, args.batch_size), lr=args.lr,
                                 device=device, p_norm=args.p_norm, p_anom=args.p_anom)
    if len(X)==0 or len(A)==0:
        raise RuntimeError("Bootstrap produced empty X or A. Adjust p_norm/p_anom or AE settings.")

    # 4) Interp train + eval
    _ = train_interp(X, A,
                     val_w, val_y,
                     hold_w, hold_y,
                     hidden=args.interp_hidden, epochs=args.interp_epochs,
                     batch=args.batch_size, lr=args.lr, lam0=args.lam0, lam1=args.lam1,
                     device=device, patience=args.patience, run_dir=args.run_dir)

    print(f"\nDone. Artifacts in: {args.run_dir}")
    print("  - best_D.pt, best_G.pt")
    print("  - val_thr_star.npy, val_pos_rate.npy")

if __name__ == "__main__":
    main()
