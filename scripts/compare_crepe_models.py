import torch
import torchaudio
import torchcrepe
import os
import json

def compare_models(fpath):
    print(f"\n--- Comparing models on: {fpath} ---")
    audio, sr = torchaudio.load(fpath)
    
    # Preprocess exactly like preprocess.py
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Signal Hardening
    audio = audio - torch.mean(audio)
    peak = torch.abs(audio).max()
    if peak > 1e-7:
        audio = audio / peak

    for model_size in ["tiny", "full"]:
        print(f"\nTesting model: {model_size}")
        f0, confidence = torchcrepe.predict(
            audio,
            sample_rate=16000,
            hop_length=160,
            fmin=50,
            fmax=2000,
            model=model_size,
            device='cpu',
            return_periodicity=True
        )
        
        ceiling_hits = (f0 >= 1600).sum().item()
        total_frames = f0.numel()
        avg_conf = confidence.mean().item()
        print(f"[{model_size}] Confidence: {avg_conf:.4f} | Hits: {ceiling_hits}/{total_frames} ({ceiling_hits/total_frames*100:.1f}%)")

if __name__ == "__main__":
    # Using one of the 100% failure files from the logs
    f = "data/raw/guitarset/05_Jazz3-150-C_comp_mic.wav"
    if os.path.exists(f):
        compare_models(f)
    else:
        print(f"File not found: {f}")
