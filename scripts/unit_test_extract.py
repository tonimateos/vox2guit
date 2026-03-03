import torch
import torchaudio
import json
import os
import sys

# Import the actual function
from preprocess import extract_features, PREPROCESS_CONFIG

def test_actual_function(fpath):
    print(f"\n--- Testing Actual extract_features on: {fpath} ---")
    audio, sr = torchaudio.load(fpath)
    
    # Preprocess like the loop
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        
    print(f"Input Tensor Shape: {audio.shape}")
    print(f"Input Tensor Peak: {audio.abs().max().item():.6f}")
    
    # Run it
    f0, loudness, confidence = extract_features(audio, 16000)
    
    ceiling_hits = (f0 >= 0.8 * PREPROCESS_CONFIG["pitch_max_freq"]).sum().item()
    print(f"Result Confidence: {confidence.mean().item():.6f}")
    print(f"Ceiling Hits: {ceiling_hits}/{f0.numel()}")

if __name__ == "__main__":
    f = "data/raw/guitarset/05_Jazz3-150-C_comp_mic.wav"
    if os.path.exists(f):
        test_actual_function(f)
    else:
        print(f"File not found: {f}")
