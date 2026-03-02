import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from core import NeuralGuitarCore

def debug_harmonics(input_wav, checkpoint_path, config_path, config_name, output_dir):
    # 1. Initialize Core
    core = NeuralGuitarCore(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        config_name=config_name
    )
    
    # 2. Process Audio and return controls
    print(f"Analyzing: {input_wav}")
    # Load audio manually to ensure we can get controls
    import librosa
    audio, sr = librosa.load(input_wav, sr=core.config["sample_rate"], mono=True)
    audio_torch = torch.from_numpy(audio).float().unsqueeze(0)
    
    with torch.no_grad():
        from preprocess import extract_features
        f0, loudness = extract_features(audio_torch, core.config["sample_rate"], hop_length=core.config["hop_length"])
        f0 = f0.to(core.device)
        loudness = loudness.to(core.device)
        
        # Get internal controls
        controls = core.model(f0, loudness, return_controls=True)
    
    harm_amps = controls['harm_amps'].squeeze(0).cpu().numpy()
    noise_mags = controls['noise_mags'].squeeze(0).cpu().numpy()
    f0_np = f0.squeeze().cpu().numpy()
    
    # 3. Visualization
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Harmonic Amplitudes Heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(harm_amps.T, aspect='auto', origin='lower', cmap='magma', interpolation='nearest')
    plt.colorbar(label='Amplitude')
    plt.title(f"Harmonic Distribution Over Time ({config_name})")
    plt.xlabel("Time (frames)")
    plt.ylabel("Harmonic Index (1-101)")
    
    # Overlay f0 (normalized for plotting)
    # Just to see if they align temporally
    ax2 = plt.gca().twinx()
    ax2.plot(f0_np, color='cyan', alpha=0.5, label='F0 (Hz)')
    ax2.set_ylabel('F0 (Hz)', color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"harmonics_{config_name}.png")
    plt.savefig(plot_path)
    print(f"Saved harmonic heatmap to: {plot_path}")
    
    # Plot 2: Noise Magnitudes
    plt.figure(figsize=(12, 4))
    plt.imshow(noise_mags.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(f"Noise Band Magnitudes Over Time ({config_name})")
    plt.xlabel("Time (frames)")
    plt.ylabel("Noise Band Index")
    plt.tight_layout()
    noise_plot_path = os.path.join(output_dir, f"noise_{config_name}.png")
    plt.savefig(noise_plot_path)
    print(f"Saved noise heatmap to: {noise_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="tests/reference_input.wav")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--config_name", type=str, default="deep")
    parser.add_argument("--output_dir", type=str, default="output/debug")
    
    args = parser.parse_args()
    debug_harmonics(args.input, args.checkpoint, args.config, args.config_name, args.output_dir)
