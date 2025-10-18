# models.py
import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, input_dim)
    def forward(self, x):
        B, T, D = x.shape
        _, (h, _) = self.enc(x)
        h0 = h[-1:].contiguous()
        dec_in = h0.transpose(0,1).repeat(1, T, 1)
        dec_out, _ = self.dec(dec_in)
        return self.out(dec_out)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, rnn="gru"):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True) if rnn.lower()=="gru" else nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h, _ = self.rnn(x)
        h_last = h[:, -1, :]
        return self.head(h_last).squeeze(-1)

class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, rnn="gru"):
        super().__init__()
        RNN = nn.GRU if rnn.lower()=="gru" else nn.LSTM
        self.enc_x = RNN(input_dim, hidden, batch_first=True)
        self.enc_a = RNN(input_dim, hidden, batch_first=True)
        self.fuse  = nn.Sequential(
            nn.Linear(hidden*2 + 1, hidden*2), nn.ReLU(inplace=True),
            nn.Linear(hidden*2, hidden), nn.ReLU(inplace=True)
        )
        self.dec   = RNN(hidden + 1, hidden, batch_first=True)
        self.out   = nn.Linear(hidden, input_dim)
    def forward(self, x, a, gamma):
        B, T, D = x.shape
        if gamma.dim() == 0:
            gamma = gamma.unsqueeze(0).repeat(B)
        hx, _ = self.enc_x(x); ha, _ = self.enc_a(a)
        hx = hx[:, -1, :]; ha = ha[:, -1, :]
        g  = gamma.view(B,1)
        z  = self.fuse(torch.cat([hx, ha, g], dim=1))
        z_rep = z.unsqueeze(1).repeat(1, T, 1)
        g_rep = g.unsqueeze(1).repeat(1, T, 1)
        dec_in = torch.cat([z_rep, g_rep], dim=2)
        dec_out, _ = self.dec(dec_in)
        return self.out(dec_out)
