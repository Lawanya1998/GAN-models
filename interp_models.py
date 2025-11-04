# interp_models.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """Regresses gamma ∈ [0,1] from a sequence [B,T,D]."""
    def __init__(self, d_in: int, h: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2*h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid()
        )

    def forward(self, seq):          # seq: [B,T,D]
        h, _ = self.lstm(seq)        # [B,T,2h]
        h = h.mean(dim=1)            # [B,2h]
        return self.head(h).squeeze(1)  # [B]

class Generator(nn.Module):
    """G(x,a,γ): concat(x,a,γ) → sequence [B,T,D]."""
    def __init__(self, d_in: int, h: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(2*d_in + 1, h, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2*h, h),
            nn.ReLU(),
            nn.Linear(h, d_in)
        )

    def forward(self, x, a, gamma):  # x,a: [B,T,D]; gamma: [B] or scalar
        B, T, D = x.shape
        if gamma.ndim == 0:
            gamma = gamma.view(1)
        g = gamma.view(-1, 1, 1).expand(B, T, 1)  # [B,T,1]
        inp = torch.cat([x, a, g], dim=-1)        # [B,T,2D+1]
        h, _ = self.lstm(inp)                     # [B,T,2h]
        return self.head(h)                       # [B,T,D]
