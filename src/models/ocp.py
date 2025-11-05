import torch, torch.nn as nn, torch.nn.functional as F

class OCPHead(nn.Module):
    """
    One-class prior head using running mean/cov of bona-fide embeddings (EMA).
    Energy score = Mahalanobis distance; lower => more bona-fide-like.
    """
    def __init__(self, feat_dim=192, momentum=0.99, eps=1e-4):
        super().__init__()
        self.register_buffer("mu", torch.zeros(feat_dim))
        self.register_buffer("cov", torch.eye(feat_dim))
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def update_stats(self, feats):
        # feats: bona-fide embeddings [N,D]
        mu_new = feats.mean(dim=0)
        self.mu = self.momentum*self.mu + (1-self.momentum)*mu_new
        # diag covariance for stability
        var_new = feats.var(dim=0) + self.eps
        self.cov = self.momentum*self.cov + (1-self.momentum)*torch.diag(var_new)

    def energy(self, z):
        # z: [B,D]
        diff = z - self.mu
        inv = torch.inverse(self.cov + self.eps*torch.eye(self.cov.size(0), device=z.device))
        e = torch.sqrt(torch.clamp(torch.sum(diff @ inv * diff, dim=1), min=0.0))
        return e  # higher => more spoof-like

    def forward(self, z):
        return self.energy(z)
