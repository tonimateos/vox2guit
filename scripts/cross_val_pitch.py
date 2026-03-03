import torch
import torchaudio
import torchcrepe
import librosa
import numpy as np
import os

def cross_validate_pitch(fpath):
    print(f"\n--- Cross-Validating: {fpath} ---")
    audio, sr = torchaudio.load(fpath)
    
    # Preprocess
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    
    audio_np = audio.squeeze(0).cpu().numpy()
    
    # 1. Librosa YIN (Heuristic)
    print("Running Librosa YIN...")
    f0_yin = librosa.yin(audio_np, fmin=50, fmax=1200, sr=16000, hop_length=160)
    yin_valid = f0_yin[f0_yin < 1200]
    print(f"[YIN] Mean Pitch: {np.mean(yin_valid):.2f} Hz | Valid Frames: {len(yin_valid)}/{len(f0_yin)}")

    # 2. CREPE (Neural)
    print("Running CREPE (tiny)...")
    f0_crepe, confidence = torchcrepe.predict(
        audio,
        sample_rate=16000,
        hop_length=160,
        fmin=50,
        fmax=1200,
        model='tiny',
        device='cpu',
        return_periodicity=True
    )
    crepe_hits = (f0_crepe >= 1100).sum().item()
    print(f"[CREPE] Avg Confidence: {confidence.mean().item():.6f}")
    print(f"[CREPE] Hits Ceiling: {crepe_hits}/{f0_crepe.numel()}")

if __name__ == "__main__":
    f = "data/raw/guitarset/05_Jazz3-150-C_comp_mic.wav"
    if os.path.exists(f):
        cross_validate_pitch(f)
    else:
        print(f"File not found: {f}")
