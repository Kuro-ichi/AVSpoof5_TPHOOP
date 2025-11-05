import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import ASVspoofDataset

class ASVspoofDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dcfg = cfg["data"]
        self.root = dcfg["root"]
        self.sample_rate = dcfg["sample_rate"]
        self.duration_sec = dcfg["duration_sec"]
        self.batch_size = dcfg["batch_size"]
        self.num_workers = dcfg.get("num_workers",4)
        self.shuffle = dcfg.get("shuffle", True)

    def setup(self, stage=None):
        self.train_ds = ASVspoofDataset(self.root, self.cfg["data"]["train_split"], self.sample_rate, self.duration_sec)
        self.dev_ds   = ASVspoofDataset(self.root, self.cfg["data"]["dev_split"], self.sample_rate, self.duration_sec)
        self.eval_ds  = ASVspoofDataset(self.root, self.cfg["data"]["eval_split"], self.sample_rate, self.duration_sec)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
