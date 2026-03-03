import torch
import torchcrepe
import numpy as np

def test_synthetic():
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    audio_np = np.sin(2 * np.pi * freq * t).astype(np.float32)
    audio = torch.from_numpy(audio_np).unsqueeze(0) # (1, Samples)
    
    print(f"--- Testing Synthetic 440Hz Sine Wave ---")
    print(f"Shape: {audio.shape}")
    
    for model_size in ["tiny", "full"]:
        try:
            f0, confidence = torchcrepe.predict(
                audio,
                sample_rate=sr,
                hop_length=160,
                fmin=50,
                fmax=2000,
                model=model_size,
                device='cpu',
                return_periodicity=True
            )
            print(f"[{model_size}] Detected Pitch: {f0.mean().item():.2f}")
            print(f"[{model_size}] Confidence: {confidence.mean().item():.6f}")
        except Exception as e:
            print(f"[{model_size}] ERROR: {e}")

if __name__ == "__main__":
    test_synthetic()
