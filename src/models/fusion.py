import torch, torch.nn as nn, torch.nn.functional as F

class EnvEmbed(nn.Module):
    def __init__(self, in_dim=4, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))

    def forward(self, stats):  # stats: [B,4] (e.g., SNR, rms, zcr, dur)
        return self.net(stats)

class GatedFiLMFusion(nn.Module):
    def __init__(self, in_dims=(64,64,64), hidden=128):
        super().__init__()
        self.total = sum(in_dims)
        self.fc = nn.Sequential(nn.Linear(self.total, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(self.total, hidden), nn.Sigmoid())

    def forward(self, feats, env=None):
        x = torch.cat(feats, dim=-1)
        h = self.fc(x)
        g = self.gate(x)
        return h * g  # [B,hidden]
