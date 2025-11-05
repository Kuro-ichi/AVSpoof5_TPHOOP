# ASVspoof5 TPH-OP Skeleton

This is a **production-ready skeleton** for the paper-style architecture you requested:
**TPH-OP (Tri-Path Hybrid with One-class Prior)** for ASVspoof 5 (CM) with optional SASV fusion.

> Goal: strong generalization to **unseen attacks**, robust to device/room/codec shifts.

## Contents
- `src/models/branches.py` — three complementary branches:
  - `RawCNNBranch` (raw waveform)
  - `SpectroPhaseBranch` (mel + phase/group-delay, Conformer-ish encoder)
  - `SSLBranch` (self-supervised speech features; configurable)
- `src/models/fusion.py` — Gated FiLM fusion conditioned on env-embedding.
- `src/models/ocp.py` — One-class prior head (Mahalanobis energy, bona-fide only).
- `src/models/model.py` — LightningModule combining losses, calibration-ready output.
- `src/data/dataset.py`, `src/data/datamodule.py` — dataset & datamodule (ASVspoof-like).
- `src/utils/metrics.py` — EER & min-tDCF (simple reference implementation).
- `src/utils/audio.py` — feature helpers (STFT/mel/phase, augment stubs).
- `src/train.py` — entry point (Lightning).

## Quickstart
```bash
# 1) Create env (suggested)
conda create -n asvspoof5 python=3.10 -y && conda activate asvspoof5

# 2) Install deps
pip install -r requirements.txt

# 3) Adjust config
nano configs/default.yaml

# 4) Train
python -m src.train --config configs/default.yaml

# 5) Evaluate (scores + EER/min-tDCF)
python -m src.train --config configs/default.yaml --mode eval --ckpt path/to.ckpt
```

### Dataset layout (expected)
```
/path/to/data/
  train/
    wavs/xxx.wav
    protos.csv         # columns: file, label (bonafide=1/spoof=0), speaker, attack, split
  dev/
    wavs/...
    protos.csv
  eval/
    wavs/...
    protos.csv
```

You can easily adapt to official ASVspoof5 protocol CSVs (map columns in `dataset.py`).

## Notes
- SSL branch can point to any embedding extractor you like (e.g., torchaudio WavLM/XLS-R or HuggingFace). By default we keep it **frozen** and allow shallow FT. Replace encoder in `SSLBranch` as needed.
- OCP head is implemented as **class-conditional Mahalanobis** energy trained only on bona-fide batches; swap with a normalizing flow if desired.
- Test-Time Adaptation (TTA) hooks are left as TODO stubs.
- min-tDCF implementation here is a compact reference; for leaderboard you must use the official scorer.

Happy hacking!
