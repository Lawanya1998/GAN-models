# train_tsgenad.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_tsgenad import *
from data import NoBoomDataset
from interp_data import build_X_A_from_train, PairInterpDataset, score_test_windows
from interp_models import Generator, Discriminator

def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1) Load data
    train_ds = NoBoomDataset(DATASET, VERSION, ROOT, train=True,
                             include_misc_faults=True, include_controller_faults=True)
    test_ds  = NoBoomDataset(DATASET, VERSION, ROOT, train=False,
                             include_misc_faults=True, include_controller_faults=True)
    D_feat = train_ds.num_features
    print(f"[Data] features={D_feat}")

    # 2) Make windows + split X/A from TRAIN
    X_np, A_np = build_X_A_from_train(train_ds, T, STRIDE)
    if len(X_np) == 0 or len(A_np) == 0:
        raise RuntimeError(f"Need normals & anomalies. Got X={len(X_np)}, A={len(A_np)}. T={T}, STRIDE={STRIDE}")
    print(f"[Prep] X={len(X_np)}, A={len(A_np)}, T={T}, D={D_feat}")

    ds_pairs = PairInterpDataset(X_np, A_np)
    loader   = DataLoader(ds_pairs, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3) Models
    G = Generator(D_feat, HIDDEN).to(DEVICE)
    D = Discriminator(D_feat, HIDDEN).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=LR_G)
    optD = torch.optim.Adam(D.parameters(), lr=LR_D)
    mse  = nn.MSELoss()

    # 4) Train
    for epoch in range(1, EPOCHS+1):
        G.train(); D.train()
        lossD, lossG, nD, nG = 0.0, 0.0, 0, 0

        for x, a, gamma in loader:
            x = x.to(DEVICE).float()
            a = a.to(DEVICE).float()
            gamma = gamma.to(DEVICE).float()

            # D step: regress gamma on G(x,a,gamma)
            optD.zero_grad()
            with torch.no_grad():
                yg = G(x, a, gamma)
            pred = D(yg)
            dloss = mse(pred, gamma)
            dloss.backward()
            optD.step()
            lossD += dloss.item(); nD += 1

            # G step: boundary reconstruction at gamma=0 and gamma=1
            optG.zero_grad()
            y0 = G(x, a, torch.zeros_like(gamma))  # ~ x
            y1 = G(x, a, torch.ones_like(gamma))   # ~ a
            gloss = LAMBDA0 * mse(y0, x) + LAMBDA1 * mse(y1, a)
            gloss.backward()
            optG.step()
            lossG += gloss.item(); nG += 1

        print(f"[Epoch {epoch:02d}] D_loss={lossD/max(nD,1):.4f} | G_loss={lossG/max(nG,1):.4f}")

    # 5) Save checkpoints
    torch.save(D.state_dict(), os.path.join(RUN_DIR, D_CKPT))
    torch.save(G.state_dict(), os.path.join(RUN_DIR, G_CKPT))
    print(f"[Save] Checkpoints saved to {RUN_DIR}/")

    # 6) Quick scoring demo on TEST
    wins, scores = score_test_windows(test_ds, D, T, STRIDE, DEVICE)
    print(f"[Eval] Scored {len(scores)} test windows. Sample scores: {scores[:10]}")

if __name__ == "__main__":
    main()
