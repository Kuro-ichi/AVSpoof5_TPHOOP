import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from src.models.branches import RawCNNBranch, SpectroPhaseBranch, SSLBranch
from src.models.fusion import GatedFiLMFusion
from src.models.ocp import OCPHead
from src.utils.metrics import compute_eer, compute_min_tdcf

class TPHOPSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        mc = cfg["model"]
        self.branches = nn.ModuleList()
        dims = []
        if mc["raw_branch"]["enabled"]:
            self.raw = RawCNNBranch(tuple(mc["raw_branch"].get("channels",[16,32,64])))
            self.branches.append(self.raw); dims.append(64)
        if mc["spectro_phase_branch"]["enabled"]:
            sp = mc["spectro_phase_branch"]
            self.sp = SpectroPhaseBranch(mel_bins=sp["mel_bins"], d_model=sp["d_model"], n_layers=sp["n_layers"], n_heads=sp["n_heads"])
            self.branches.append(self.sp); dims.append(64)
        if mc["ssl_branch"]["enabled"]:
            self.ssl = SSLBranch(proj_dim=mc["ssl_branch"]["proj_dim"])
            self.branches.append(self.ssl); dims.append(64)

        self.fusion = GatedFiLMFusion(in_dims=tuple(dims), hidden=mc["fusion"]["hidden"])
        self.cls = nn.Linear(mc["fusion"]["hidden"], 1)  # bona-fide prob
        self.sigmoid = nn.Sigmoid()

        if mc["ocp"]["enabled"]:
            self.ocp = OCPHead(feat_dim=mc["fusion"]["hidden"], momentum=mc["ocp"]["momentum"])
        else:
            self.ocp = None

        self.lr = cfg["optim"]["lr"]
        self.wd = cfg["optim"]["weight_decay"]

        self.train_preds, self.train_labels = [], []
        self.dev_scores, self.dev_labels = [], []
        self.test_scores, self.test_labels = [], []

    def forward(self, wav):
        feats = [b(wav) for b in self.branches]
        z = self.fusion(feats, env=None)  # [B,H]
        logit = self.cls(z).squeeze(-1)   # higher => bona-fide
        out = {"logit": logit, "embed": z}
        if self.ocp is not None:
            out["energy"] = self.ocp(z)
        return out

    def training_step(self, batch, batch_idx):
        wav, y, meta = batch  # y in {0,1} where 1=bonafide
        out = self(wav)
        bce = F.binary_cross_entropy_with_logits(out["logit"], y)
        loss = bce
        # update OCP with bona-fide only
        if self.ocp is not None:
            with torch.no_grad():
                bf_mask = (y > 0.5)
                if bf_mask.any():
                    self.ocp.update_stats(out["embed"][bf_mask])
            e = out["energy"]
            # map energy to spoof-prob target (higher energy => spoof)
            ocp_loss = F.mse_loss((e - e.min())/(e.max()-e.min()+1e-6), (1.0 - y))
            loss = loss + self.hparams["loss"]["ocp_weight"]*ocp_loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            p = torch.sigmoid(out["logit"])
            self.train_preds.append(p.detach().cpu())
            self.train_labels.append(y.detach().cpu())
        return loss

    def on_validation_epoch_start(self):
        self.dev_scores, self.dev_labels = [], []

    def validation_step(self, batch, batch_idx):
        wav, y, meta = batch
        out = self(wav)
        score = torch.sigmoid(out["logit"]).detach().cpu().numpy()
        self.dev_scores.extend(score.tolist())
        self.dev_labels.extend(y.cpu().numpy().tolist())

    def on_validation_epoch_end(self):
        import numpy as np
        scores = np.array(self.dev_scores); labels = np.array(self.dev_labels)
        eer = compute_eer(scores, labels)
        min_tdcf = compute_min_tdcf(scores, labels)
        self.log("dev/eer", eer, prog_bar=True)
        self.log("dev/min_tdcf", min_tdcf)

    def on_test_epoch_start(self):
        self.test_scores, self.test_labels = [], []

    def test_step(self, batch, batch_idx):
        wav, y, meta = batch
        out = self(wav)
        score = torch.sigmoid(out["logit"]).detach().cpu().numpy()
        self.test_scores.extend(score.tolist())
        self.test_labels.extend(y.cpu().numpy().tolist())

    def on_test_epoch_end(self):
        import numpy as np
        scores = np.array(self.test_scores); labels = np.array(self.test_labels)
        eer = compute_eer(scores, labels)
        min_tdcf = compute_min_tdcf(scores, labels)
        self.log("eval/eer", eer, prog_bar=True)
        self.log("eval/min_tdcf", min_tdcf)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams["optim"]["max_epochs"])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval":"epoch"}}
