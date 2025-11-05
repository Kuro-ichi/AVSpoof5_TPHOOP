import os, csv, torch
import torchaudio
from torch.utils.data import Dataset
import random
from src.utils.audio import load_wav_fixed, compute_features

class ASVspoofDataset(Dataset):
    def __init__(self, root, split, sample_rate=16000, duration_sec=6.0):
        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.duration = int(duration_sec * sample_rate)
        self.wav_dir = os.path.join(root, split, "wavs")
        self.proto = os.path.join(root, split, "protos.csv")
        self.items = []
        with open(self.proto, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append({
                    "file": row["file"],
                    "label": int(row["label"]),
                    "speaker": row.get("speaker","unknown"),
                    "attack": row.get("attack","unknown")
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path = os.path.join(self.wav_dir, it["file"])
        wav = load_wav_fixed(path, self.sample_rate, self.duration)  # [T]
        # basic features computed on-the-fly inside model branches; return raw
        y = torch.tensor(float(it["label"]), dtype=torch.float32)
        meta = {"speaker": it["speaker"], "attack": it["attack"], "file": it["file"]}
        return wav, y, meta
