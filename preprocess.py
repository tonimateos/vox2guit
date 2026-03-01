import os
import glob
import torch
import torchaudio
import torchcrepe
import numpy as np
from tqdm import tqdm
import argparse

# ==============================================================================
# Research Note: Data Pipeline & Feature Extraction
# ==============================================================================
# 1. Timbre Transfer requires tight alignment between Input Features (Control)
#    and Target Audio.
# 2. Pitch Extraction (CREPE): We use CREPE (Convolutional Representation for 
#    Pitch Estimation) because it is state-of-the-art for monophonic pitch 
#    tracking and robust to noise/reverb, unlike heuristic methods (YIN).
# 3. Loudness extraction: We compute A-weighted loudness or simple RMS. 
#    This forms the "Volume Envelope" control signal.
# 4. Sampling Rate: 16kHz is standard for DDSP to balance quality/speed.
# ==============================================================================

import scipy.signal

def a_weighting_filter(audio: torch.Tensor, sample_rate: int):
    """
    Applies A-weighting filter to the audio signal using stable Second-Order Sections (SOS).
    """
    if sample_rate != 16000:
        raise ValueError("A-weighting filter tuned for 16kHz.")
    
    # Design A-weighting filter directly in digital domain as SOS
    # Standard frequencies (approximate Butterworth approximation for A-weighting)
    # Note: Upper cutoff capped at 7500 to stay below Nyquist (8000) for 16kHz SR.
    sos = scipy.signal.iirfilter(6, [20.6, 7500], rs=60, btype='bandpass', 
                                  analog=False, ftype='butter', fs=sample_rate, output='sos')
    
    audio_np = audio.numpy()
    filtered_audio = scipy.signal.sosfilt(sos, audio_np, axis=-1)
    return torch.from_numpy(filtered_audio.copy()).float()

def extract_features(audio: torch.Tensor, sample_rate: int, hop_length: int = 160, existing_f0: torch.Tensor = None):
    """
    Extract f0 and loudness from audio.
    """
    # 1. Extract A-Weighted Loudness
    weighted_audio = a_weighting_filter(audio, sample_rate)
    frame_length = 1024
    audio_pad = torch.nn.functional.pad(weighted_audio, (frame_length // 2, frame_length // 2))
    audio_frames = audio_pad.unfold(1, frame_length, hop_length) 
    loudness = torch.sqrt(torch.mean(audio_frames**2, dim=-1) + 1e-7) 
    loudness = loudness.unsqueeze(-1) 
    
    # 2. Extract or Reuse Pitch
    if existing_f0 is not None:
        f0 = existing_f0
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        audio_16k = audio # Assuming SR=16k
        f0, _ = torchcrepe.predict(
            audio_16k, sample_rate=16000, hop_length=hop_length, fmin=50, fmax=2000, 
            model='tiny', batch_size=2048, device=device, return_periodicity=True
        )
        f0 = f0.unsqueeze(-1)
    
    # Match lengths
    min_len = min(f0.shape[1], loudness.shape[1])
    f0 = f0[:, :min_len, :]
    loudness = loudness[:, :min_len, :]
    
    return f0, loudness


def preprocess_dataset(input_dir: str, output_dir: str, hop_length: int = 160):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, '*.wav'))
    
    print(f"Found {len(files)} wav files.")
    
    import warnings
    warnings.filterwarnings("ignore") # Ignore some torchaudio warnings

    for fpath in tqdm(files):
        fname = os.path.basename(fpath).replace('.wav', '.pt')
        save_path = os.path.join(output_dir, fname)
        
        existing_f0 = None
        if os.path.exists(save_path):
            try:
                d = torch.load(save_path)
                if not torch.isnan(d['loudness']).any():
                    continue
                # Reuse existing f0 to avoid slow CREPE re-inference
                existing_f0 = d['f0']
                print(f"Repairing loudness for: {fname}")
            except:
                pass

        # Load
        audio, sr = torchaudio.load(fpath)
        # Mix to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample to 16k
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
            
        # Extract
        try:
            f0, loudness = extract_features(audio, 16000, hop_length=hop_length, existing_f0=existing_f0)
            
            # Save
            num_frames = f0.shape[1]
            audio_target = audio[:, :num_frames * 160]
            
            torch.save({
                'f0': f0.cpu(),
                'loudness': loudness.cpu(),
                'audio': audio_target.cpu()
            }, save_path)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of .wav files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save .pt tensors')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--config_name', type=str, default='tiny')
    args = parser.parse_args()
    
    import json
    with open(args.config_file, "r") as f:
        config = json.load(f)[args.config_name]
    
    preprocess_dataset(args.input_dir, args.output_dir, hop_length=config['hop_length'])
