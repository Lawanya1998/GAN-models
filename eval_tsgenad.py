# eval_tsgenad.py
import os
import numpy as np
import torch
import pandas as pd

from config_tsgenad import *
from data import NoBoomDataset
from interp_data import score_test_windows
from interp_models import Discriminator

def main():
    test_ds = NoBoomDataset(DATASET, VERSION, ROOT, train=False,
                            include_misc_faults=True, include_controller_faults=True)
    D_feat = test_ds.num_features

    D = Discriminator(D_feat, HIDDEN).to(DEVICE)
    D.load_state_dict(torch.load(os.path.join(RUN_DIR, D_CKPT), map_location=DEVICE))
    D.eval()

    wins, scores = score_test_windows(test_ds, D, T, STRIDE, DEVICE)
    print(f"[Eval] {len(scores)} windows; mean={scores.mean():.4f} std={scores.std():.4f}")

    out_csv = os.path.join(RUN_DIR, "test_window_scores.csv")
    pd.DataFrame({"score": scores}).to_csv(out_csv, index=False)
    print(f"[Save] Scores → {out_csv}")

if __name__ == "__main__":
    main()
