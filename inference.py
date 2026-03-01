import os
import argparse
import torch
import torchaudio
import json
from model import NeuralGuitar
from preprocess import extract_features

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load external config
    with open(args.config_file, "r") as f:
        all_configs = json.load(f)
    net_config = all_configs[args.config_name]

    # 1. Load Model
    model = NeuralGuitar(config=net_config).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Robust loading (handle both state_dict and full checkpoint)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 2. Get Features (f0, loudness)
    if args.input_pt:
        print(f"Loading features from: {args.input_pt}")
        data = torch.load(args.input_pt)
        f0 = data['f0'].to(device)
        loudness = data['loudness'].to(device)
    elif args.input_wav:
        print(f"Extracting features from: {args.input_wav}")
        audio, sr = torchaudio.load(args.input_wav)
        # Mix to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample to 16k
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        with torch.no_grad():
            f0, loudness = extract_features(audio, 16000, hop_length=net_config['hop_length'])
            f0 = f0.to(device)
            loudness = loudness.to(device)
    else:
        raise ValueError("Must provide either --input_wav or --input_pt")

    # 3. Model Forward
    print("Synthesizing...")
    with torch.no_grad():
        output_audio = model(f0, loudness)
    
    # 4. Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = os.path.basename(args.input_wav or args.input_pt).split('.')[0] + "_resynth.wav"
    out_path = os.path.join(args.output_dir, out_name)
    
    # output_audio is [Batch, Time], torchaudio expects [Channels, Time]
    torchaudio.save(out_path, output_audio.cpu(), 16000)
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
