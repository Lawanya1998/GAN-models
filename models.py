# models.py
# models.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Sequence encoder (GRU/LSTM) + MLP head -> predicts a scalar in [0,1].
    We will train it as a regressor of gamma (interpolation factor).
    Later, we use D(x) as an anomaly score proxy on real windows.
    """
    def __init__(self, input_dim: int, hidden: int = 64, rnn: str = "gru"):
        super().__init__()
        rnn = rnn.lower()
        RNN = nn.GRU if rnn == "gru" else nn.LSTM
        self.rnn = RNN(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, _ = self.rnn(x)            # [B,T,H]
        h_last = h[:, -1, :]          # [B,H]
        return self.head(h_last).squeeze(-1)  # [B]

class Generator(nn.Module):
    """
    Encodes a normal window (x) and an anomaly window (a).
    Fuses their states + gamma, then decodes a sequence y = G(x,a,gamma).
    Enforces anchors: gamma=0 -> y≈x, gamma=1 -> y≈a.
    """
    def __init__(self, input_dim: int, hidden: int = 64, rnn: str = "gru"):
        super().__init__()
        rnn = rnn.lower()
        RNN = nn.GRU if rnn == "gru" else nn.LSTM
        self.enc_x = RNN(input_dim, hidden, batch_first=True)
        self.enc_a = RNN(input_dim, hidden, batch_first=True)
        self.fuse  = nn.Sequential(
            nn.Linear(hidden*2 + 1, hidden*2), nn.ReLU(inplace=True),
            nn.Linear(hidden*2, hidden), nn.ReLU(inplace=True)
        )
        self.dec   = RNN(hidden + 1, hidden, batch_first=True)
        self.out   = nn.Linear(hidden, input_dim)

    def forward(self, x, a, gamma):
        """
        x, a: [B,T,D]  gamma: [B] (0..1)
        """
        B, T, D = x.shape
        if gamma.dim() == 0:
            gamma = gamma.unsqueeze(0).repeat(B)
        hx, _ = self.enc_x(x)
        ha, _ = self.enc_a(a)
        hx = hx[:, -1, :]     # [B,H]
        ha = ha[:, -1, :]     # [B,H]
        g  = gamma.view(B, 1) # [B,1]
        z  = self.fuse(torch.cat([hx, ha, g], dim=1))   # [B,H]
        z_rep = z.unsqueeze(1).repeat(1, T, 1)          # [B,T,H]
        g_rep = g.unsqueeze(1).repeat(1, T, 1)          # [B,T,1]
        dec_in = torch.cat([z_rep, g_rep], dim=2)       # [B,T,H+1]
        dec_out, _ = self.dec(dec_in)                   # [B,T,H]
        return self.out(dec_out)                        # [B,T,D]
