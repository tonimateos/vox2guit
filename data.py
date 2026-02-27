import os
import torch
from torch.utils.data import Dataset
import glob

class NeuralGuitarDataset(Dataset):
    """
    Dataset for loading preprocessed Neural Guitar features (f0, loudness, audio).
    """
    def __init__(self, data_dir: str, sequence_length: int = 16000):
        """
        Args:
            data_dir (str): Directory containing .pt files.
            sequence_length (int): Length of audio segments in samples (Default: 1s at 16k).
        """
        self.files = glob.glob(os.path.join(data_dir, '*.pt'))
        self.sequence_length = sequence_length
        self.hop_length = 160 # Matches preprocess.py
        self.frames_per_seq = sequence_length // self.hop_length
        
        if len(self.files) == 0:
            print(f"[WARNING] No .pt files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        
        f0 = data['f0'] # [1, Frames, 1]
        loudness = data['loudness'] # [1, Frames, 1]
        audio = data['audio'] # [1, Samples]
        
        # Random Crop
        total_frames = f0.shape[1]
        if total_frames > self.frames_per_seq:
            start_frame = torch.randint(0, total_frames - self.frames_per_seq, (1,)).item()
            end_frame = start_frame + self.frames_per_seq
            
            f0 = f0[:, start_frame:end_frame, :]
            loudness = loudness[:, start_frame:end_frame, :]
            
            start_sample = start_frame * self.hop_length
            end_sample = start_sample + self.sequence_length
            audio = audio[:, start_sample:end_sample]
        
        return {
            'f0': f0.squeeze(0), # [Frames, 1]
            'loudness': loudness.squeeze(0), # [Frames, 1]
            'audio': audio.squeeze(0) # [Samples]
        }
