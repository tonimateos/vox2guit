import torch
import numpy as np
import scipy.io.wavfile as wavfile
from core import NeuralGuitarCore
import os

def generate_perfect_test():
    print("--- Generating Perfect Feature Test (C4 -> C5) ---")
    
    # 1. Initialize Core
    core = NeuralGuitarCore()
    sr = core.config["sample_rate"]
    hop_length = core.config["hop_length"]
    
    # 2. Define Note Parameters
    c4 = 261.63
    c5 = 523.25
    duration_per_note = 1.0 # second
    
    # Calculate frames
    frames_per_note = int(duration_per_note * sr / hop_length)
    total_frames = frames_per_note * 2
    
    # 3. Create Synthetic Features
    # F0: C4 then C5
    f0 = torch.zeros(1, total_frames, 1)
    f0[0, :frames_per_note, 0] = c4
    f0[0, frames_per_note:, 0] = c5
    
    # Loudness: Steady high volume (0.5)
    loudness = torch.ones(1, total_frames, 1) * 0.5
    
    # Move to device
    f0 = f0.to(core.device)
    loudness = loudness.to(core.device)
    
    # 4. Synthesize
    print("Synthesizing from perfect features...")
    with torch.no_grad():
        output_audio = core.model(f0, loudness)
        
        # Peak normalization
        max_val = torch.max(torch.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val
            
    # 5. Save
    os.makedirs("output/debug", exist_ok=True)
    out_path = "output/debug/perfect_c_octave.wav"
    audio_np = output_audio.squeeze().cpu().numpy()
    wavfile.write(out_path, sr, audio_np)
    
    # Save the features as well for inspection if needed
    torch.save({'f0': f0.cpu(), 'loudness': loudness.cpu()}, "output/debug/perfect_features.pt")
    
    print(f"✅ Success! Saved perfect synthesis to: {out_path}")
    print(f"You can now listen to this to hear if the GRU handles pure notes well.")

if __name__ == "__main__":
    generate_perfect_test()
