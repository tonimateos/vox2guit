import torch
import torchaudio
import os

def export_diagnostic(fpath):
    print(f"\n--- Exporting Diagnostic Audio for: {fpath} ---")
    audio, sr = torchaudio.load(fpath)
    
    # 1. Mono Mix
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # 2. Resample
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # 3. Normalize
    audio = audio - torch.mean(audio)
    peak = torch.abs(audio).max()
    if peak > 1e-7:
        audio = audio / peak
    
    # Save the first 5 seconds
    slice_len = 5 * target_sr
    audio_slice = audio[:, :slice_len]
    
    os.makedirs("data/diagnostic_wav", exist_ok=True)
    out_path = "data/diagnostic_wav/DEBUG_SIGNAL.wav"
    torchaudio.save(out_path, audio_slice, target_sr)
    print(f"Saved 5s diagnostic clip to: {out_path}")

if __name__ == "__main__":
    f = "data/raw/guitarset/05_Jazz3-150-C_comp_mic.wav"
    if os.path.exists(f):
        export_diagnostic(f)
    else:
        print(f"File not found: {f}")
