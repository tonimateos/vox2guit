import torch
import torchaudio
import os
import json

def check_file(fpath):
    print(f"\n--- Analyzing: {fpath} ---")
    audio, sr = torchaudio.load(fpath)
    print(f"Original Shape: {audio.shape}, SR: {sr}")
    print(f"Original Peak: {audio.abs().max().item():.6f}")
    print(f"Original RMS: {torch.sqrt(torch.mean(audio**2)).item():.6f}")
    
    # Check mono mix
    if audio.shape[0] > 1:
        mono = torch.mean(audio, dim=0, keepdim=True)
        print(f"Mono Peak: {mono.abs().max().item():.6f}")
        print(f"Mono RMS: {torch.sqrt(torch.mean(mono**2)).item():.6f}")
        if mono.abs().max() < audio.abs().max() * 0.1:
            print("[!] WARNING: Significant signal loss during mono mixing. Possible phase cancellation!")
    else:
        mono = audio

    # Check Resample
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        resampled = resampler(mono)
        print(f"Resampled Peak: {resampled.abs().max().item():.6f}")
        print(f"Resampled RMS: {torch.sqrt(torch.mean(resampled**2)).item():.6f}")

if __name__ == "__main__":
    files = [
        "data/raw/guitarset/05_Jazz3-150-C_comp_mic.wav",
        "data/raw/guitarset/03_Funk1-97-C_solo_mic.wav"
    ]
    for f in files:
        if os.path.exists(f):
            check_file(f)
        else:
            print(f"File not found: {f}")
