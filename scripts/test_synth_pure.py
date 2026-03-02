import torch
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import sys

# Allow running from scripts/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synth import HarmonicSynthesizer, FilteredNoiseSynthesizer

def test_pure_synthesis(output_dir="output/test_pure"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Constants
    sr = 16000
    duration_sec = 2.0
    hop_length = 160
    n_frames = int(duration_sec * sr / hop_length)
    n_samples = n_frames * hop_length
    
    # 1. Create Artificial Parameters
    # Pitch: Constant 220Hz (A3)
    f0 = torch.ones(1, n_frames, 1) * 220.0
    
    # Harmonic Amplitudes: Simple Sawtooth-like decay (1/n)
    # 101 harmonics
    n_harmonics = 101
    harm_amps = torch.zeros(1, n_frames, n_harmonics)
    for i in range(n_harmonics):
        # Amplitudes decay as 1/(i+1)
        harm_amps[:, :, i] = 1.0 / (i + 1)
        
    # IMPORTANT: Normalize so that for any frame, the sum of amplitudes is 1.0
    # This matches the Softmax behavior in our NeuralGuitar model and prevents clipping.
    harm_amps = harm_amps / harm_amps.sum(dim=-1, keepdim=True)
        
    # Apply a global exponential decay envelope to the whole note
    time_frames = torch.linspace(0, duration_sec, n_frames)
    envelope = torch.exp(-2.0 * time_frames).unsqueeze(0).unsqueeze(-1)
    harm_amps = harm_amps * envelope
    
    # Noise: Short burst at the beginning (the "pick") then silence
    n_noise_bands = 65
    noise_mags = torch.zeros(1, n_frames, n_noise_bands)
    # Burst for the first 0.1 seconds
    burst_len = int(0.1 * sr / hop_length)
    noise_mags[:, :burst_len, :] = 0.5
    
    # 2. Initialize Synthesizers
    harm_synth = HarmonicSynthesizer(n_harmonics=n_harmonics, sample_rate=sr, hop_length=hop_length)
    noise_synth = FilteredNoiseSynthesizer(n_bands=n_noise_bands)
    
    # 3. Synthesize
    with torch.no_grad():
        audio_harm = harm_synth(f0, harm_amps)
        audio_noise = noise_synth(noise_mags)
        audio_final = audio_harm + audio_noise
        
    # 4. Save Audio
    import soundfile as sf
    sf.write(os.path.join(output_dir, "pure_harmonic.wav"), audio_harm[0].numpy(), sr)
    sf.write(os.path.join(output_dir, "pure_noise.wav"), audio_noise[0].numpy(), sr)
    sf.write(os.path.join(output_dir, "pure_final.wav"), audio_final[0].numpy(), sr)
    
    print(f"--- Pure Synthesis Test Complete ---")
    print(f"Saved samples to: {output_dir}")
    
    # 5. Optional: Plot the parameters we just synthesized
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(harm_amps[0, :, :20].T.numpy(), aspect='auto', origin='lower')
    plt.title("Harms (First 20)")
    plt.subplot(1, 2, 2)
    plt.plot(audio_final[0, :sr//10].numpy())
    plt.title("Waveform (Zoomed)")
    plt.savefig(os.path.join(output_dir, "diagnostic_params.png"))

if __name__ == "__main__":
    test_pure_synthesis()
