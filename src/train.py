import os, argparse, yaml, math
from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from src.data.datamodule import ASVspoofDataModule
from src.models.model import TPHOPSystem

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["train","eval"], default=None)
    p.add_argument("--ckpt", type=str, default=None)
    return p.parse_args()

def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.mode is not None:
        cfg["mode"] = args.mode
    pl.seed_everything(cfg.get("seed", 1337), workers=True)

    dm = ASVspoofDataModule(cfg)
    model = TPHOPSystem(cfg)

    logger = CSVLogger(cfg.get("log_dir","runs"), name="tphop")
    ckpt_cb = ModelCheckpoint(monitor="dev/eer", mode="min", save_top_k=3, filename="tphop-{epoch:02d}-{dev_eer:.4f}")
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg["optim"]["max_epochs"],
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0
    )

    if cfg["mode"] == "train":
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        if args.ckpt is None:
            raise ValueError("Please provide --ckpt for eval mode.")
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt)

if __name__ == "__main__":
    main()
