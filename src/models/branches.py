import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

class RawCNNBranch(nn.Module):
    """Small raw waveform CNN to capture micro artefacts."""
    def __init__(self, channels=(16,32,64)):
        super().__init__()
        c = list(channels)
        layers = []
        in_ch = 1
        for i, ch in enumerate(c):
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=9, stride=2, padding=4, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, 64)

    def forward(self, x):  # x: [B,T]
        x = x.unsqueeze(1)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        e = self.proj(h)
        return e  # [B,64]

class ConformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, conv_expansion=2):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model*4), nn.GLU(), nn.Linear(d_model*2, d_model))
        self.mha = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model*conv_expansion, 3, padding=1, groups=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.ff2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5*self.ff1(x)
        x = x + self.mha(x, x, x, need_weights=False)[0]
        x = x + self.conv(x.transpose(1,2)).transpose(1,2)
        x = x + 0.5*self.ff2(x)
        return self.ln(x)

class SpectroPhaseBranch(nn.Module):
    """Mel + phase (instantaneous) encoder with tiny Conformer stack."""
    def __init__(self, mel_bins=80, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=mel_bins)
        self.spec_bn = nn.BatchNorm1d(mel_bins*2)
        self.proj = nn.Linear(mel_bins*2, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(d_model, 64)

    def forward(self, x):  # [B,T]
        B = x.size(0)
        S = self.mel(x) + 1e-6
        S = torch.log(S)
        # crude phase proxy: delta along time
        phase_like = torch.cat([torch.zeros(B,1,S.size(-1), device=S.device), S[:,1:,:]-S[:,:-1,:]], dim=1)
        F = torch.cat([S, phase_like], dim=1)  # [B, 2*mel, T']
        F = self.spec_bn(F)
        F = F.transpose(1,2)  # [B,T',2*mel]
        H = self.proj(F)
        for blk in self.blocks:
            H = blk(H)
        Hm = H.mean(dim=1)
        return self.out(Hm)  # [B,64]

class SSLBranch(nn.Module):
    """Placeholder SSL encoder: use torchaudio 'wav2vec2' or 'hubert' features if available."""
    def __init__(self, proj_dim=128):
        super().__init__()
        # keep it simple: a small 1D conv stack mimicking SSL features (no external weights)
        self.fe = nn.Sequential(
            nn.Conv1d(1, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, 64)

    def forward(self, x):
        h = self.fe(x.unsqueeze(1)).squeeze(-1)
        return self.proj(h)  # [B,64]
