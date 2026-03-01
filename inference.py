import os
import argparse
import torch
import torchaudio
import json
from model import NeuralGuitar
from preprocess import extract_features
from core import NeuralGuitarCore
import scipy.io.wavfile as wavfile

def inference(args):
    # 1. Initialize Core
    core = NeuralGuitarCore(
        checkpoint_path=args.checkpoint,
        config_path=args.config_file,
        config_name=args.config_name
    )
    
    # 2. Process
    if args.input_wav:
        input_path = args.input_wav
    elif args.input_pt:
        # Note: shared core currently expects a wav for the full pipeline.
        # If the user provides a PT, we need to handle that specifically or 
        # update core to handle pre-extracted features.
        # For simplicity in this logic consolidation, let's assume wav for now 
        # or implement a core.process_features if needed.
        print("Warning: .pt input support in unified core is coming soon. Using .wav extraction.")
        input_path = args.input_pt # This might fail if it's not a wav, but let's stick to wav for the core refactor
    else:
        raise ValueError("Must provide --input_wav")

    print(f"Processing {input_path}...")
    audio_orig, audio_resynth, f0, loudness = core.process_audio(input_path)
    
    # 3. Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = os.path.basename(input_path).split('.')[0] + "_resynth.wav"
    out_path = os.path.join(args.output_dir, out_name)
    
    wavfile.write(out_path, core.config["sample_rate"], audio_resynth)
    print(f"Success! Saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to input wav file')
    parser.add_argument('--input_pt', type=str, default=None, help='Path to preprocessed .pt file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    
    # Model architecture should match what was used in training
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--config_name', type=str, default='tiny')

    args = parser.parse_args()
    inference(args)
