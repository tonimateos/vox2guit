import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
import json

def visualize(fpath):
    print(f"Loading {fpath}...")
    data = torch.load(fpath)
    
    f0 = data['f0'].squeeze().numpy()
    loudness = data['loudness'].squeeze().numpy()
    audio = data['audio'].squeeze().numpy()
    
    # Time axes
    # Load config for SR and Hop
    with open("config.json", "r") as f:
        config = json.load(f)["tiny"]
    sr = config["sample_rate"]
    hop = config["hop_length"]
    
    time_audio = np.arange(len(audio)) / sr
    time_frames = np.arange(len(f0)) * (hop / sr)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Waveform
    axes[0].plot(time_audio, audio, color='gray', alpha=0.5)
    axes[0].set_title(f"Waveform: {os.path.basename(fpath)}")
    axes[0].set_ylabel("Amplitude")
    
    # 2. Pitch (f0)
    # Mask zeros (silence/unvoiced)
    f0_masked = f0.copy()
    f0_masked[f0_masked == 0] = np.nan
    axes[1].plot(time_frames, f0_masked, color='blue', label='f0 (Pitch)')
    axes[1].set_yscale('log')
    axes[1].set_title("Pitch trajectory (f0)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].grid(True, which='both', ls='-', alpha=0.5)
    
    # 3. Loudness
    axes[2].plot(time_frames, loudness, color='red', label='Loudness (A-Weighted)')
    axes[2].set_title("Loudness envelope")
    axes[2].set_ylabel("RMS (log scale approx)")
    axes[2].set_xlabel("Time (s)")
    
    plt.tight_layout()
    
    # Save to data/visualization/
    viz_dir = 'data/visualization'
    os.makedirs(viz_dir, exist_ok=True)
    
    out_name = os.path.join(viz_dir, os.path.basename(fpath).replace('.pt', '_viz.png'))
    plt.savefig(out_name)
    print(f"Saved visualization to {out_name}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to a .pt file')
    args = parser.parse_args()
    
    if args.file:
        visualize(args.file)
    else:
        # Pick the first one in processed
        files = glob.glob('data/processed/guitarset/*.pt')
        if files:
            visualize(files[0])
        else:
            print("No .pt files found in data/processed/guitarset/")
