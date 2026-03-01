import torch
import os
import json
import librosa
import numpy as np
from model import NeuralGuitar
from preprocess import extract_features

class NeuralGuitarCore:
    def __init__(self, checkpoint_path="checkpoints/latest.pth", config_path="config.json", config_name="tiny"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            all_configs = json.load(f)
        self.config = all_configs[config_name]
        
        # 2. Initialize Model
        self.model = NeuralGuitar(config=self.config).to(self.device)
        
        # 3. Load Checkpoint
        if os.path.exists(checkpoint_path):
            print(f"--- Loading checkpoint: {checkpoint_path} ---")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("--- Model loaded and ready! ---")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}. Running with uninitialized weights.")

    def process_audio(self, input_path):
        """
        Full pipeline: Load -> Features -> Synthesis -> Normalization
        """
        # 1. Load Audio (consistent with app.py's librosa usage)
        audio, _ = librosa.load(input_path, sr=self.config["sample_rate"], mono=True)
        audio_torch = torch.from_numpy(audio).float().unsqueeze(0)
        
        with torch.no_grad():
            # 2. Feature Extraction
            f0, loudness = extract_features(audio_torch, self.config["sample_rate"], hop_length=self.config["hop_length"])
            f0 = f0.to(self.device)
            loudness = loudness.to(self.device)
            
            # 3. Synthesis
            output_audio = self.model(f0, loudness)
            
            # 4. Peak Normalization (The "Soul" of the sound)
            max_val = torch.max(torch.abs(output_audio))
            if max_val > 0:
                output_audio = output_audio / max_val
                
        return audio, output_audio.squeeze().cpu().numpy(), f0, loudness
