import torch, torchaudio

def load_wav_fixed(path, sr, length):
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    wav = wav.mean(dim=0)  # mono
    if wav.numel() >= length:
        wav = wav[:length]
    else:
        pad = length - wav.numel()
        wav = torch.nn.functional.pad(wav, (0,pad))
    return wav

def compute_features(wav, sr=16000):
    # placeholder if you want to precompute; branches compute internally
    return wav
