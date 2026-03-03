import torch
import torchaudio
import torchcrepe
import os

def test_crepe(fpath):
    print(f"\n--- Testing CREPE on: {fpath} ---")
    audio, sr = torchaudio.load(fpath)
    
    # Preprocess exactly like preprocess.py
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Try Tiny vs Full
    for model_size in ["tiny", "full"]:
        print(f"\nTesting model: {model_size}")
        f0, confidence = torchcrepe.predict(
            audio.squeeze(0), # Squeeze to (Samples,) just in case
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
        print(f"[{model_size}] Hits: {ceiling_hits}/{total_frames} ({ceiling_hits/total_frames*100:.1f}%)")
        print(f"[{model_size}] Avg Confidence: {avg_conf:.6f}")

if __name__ == "__main__":
    f = "data/raw/guitarset/03_Funk1-97-C_solo_mic.wav"
    if os.path.exists(f):
        test_crepe(f)
    else:
        print(f"File not found: {f}")
