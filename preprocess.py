import os
import glob
import torch
import torchaudio
import torchcrepe
import numpy as np
from tqdm import tqdm
import argparse
import json

# --- Load Preprocessing Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_preprocess.json")
with open(CONFIG_PATH, "r") as f:
    PREPROCESS_CONFIG = json.load(f)

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
    if sample_rate != PREPROCESS_CONFIG["sample_rate"]:
        raise ValueError(f"A-weighting filter tuned for {PREPROCESS_CONFIG['sample_rate']}Hz.")
    
    # Design A-weighting filter directly in digital domain as SOS
    sos = scipy.signal.iirfilter(
        PREPROCESS_CONFIG["a_weighting_filter_order"], 
        [PREPROCESS_CONFIG["a_weighting_low_cutoff"], PREPROCESS_CONFIG["a_weighting_high_cutoff"]], 
        rs=PREPROCESS_CONFIG["a_weighting_stopband_attenuation"], 
        btype='bandpass', 
        analog=False, 
        ftype='butter', 
        fs=sample_rate, 
        output='sos'
    )
    
    audio_np = audio.numpy()
    filtered_audio = scipy.signal.sosfilt(sos, audio_np, axis=-1)
    return torch.from_numpy(filtered_audio.copy()).float()

def extract_features(audio: torch.Tensor, sample_rate: int, hop_length: int = None, existing_f0: torch.Tensor = None):
    """
    Extract f0 and loudness from audio.
    """
    hop_length = hop_length or PREPROCESS_CONFIG["hop_length"]
    
    # 1. Extract A-Weighted Loudness
    weighted_audio = a_weighting_filter(audio, sample_rate)
    frame_length = PREPROCESS_CONFIG["frame_length"]
    audio_pad = torch.nn.functional.pad(weighted_audio, (frame_length // 2, frame_length // 2))
    audio_frames = audio_pad.unfold(1, frame_length, hop_length) 
    loudness = torch.sqrt(torch.mean(audio_frames**2, dim=-1) + PREPROCESS_CONFIG["epsilon"]) 
    loudness = loudness.unsqueeze(-1) 
    
    # 2. Extract or Reuse Pitch
    if existing_f0 is not None:
        f0 = existing_f0
    else:
        # Check for CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
        audio_16k = audio # Assuming SR=16k
        f0, _ = torchcrepe.predict(
            audio_16k, 
            sample_rate=PREPROCESS_CONFIG["sample_rate"], 
            hop_length=hop_length, 
            fmin=PREPROCESS_CONFIG["pitch_min_freq"], 
            fmax=PREPROCESS_CONFIG["pitch_max_freq"], 
            model=PREPROCESS_CONFIG["crepe_model_size"], 
            batch_size=PREPROCESS_CONFIG["crepe_batch_size"], 
            device=device, 
            return_periodicity=True
        )
        f0 = f0.unsqueeze(-1)
    
    # Match lengths
    min_len = min(f0.shape[1], loudness.shape[1])
    f0 = f0[:, :min_len, :]
    loudness = loudness[:, :min_len, :]
    
    return f0, loudness


def preprocess_dataset(input_dir: str, output_dir: str, hop_length: int = None):
    hop_length = hop_length or PREPROCESS_CONFIG["hop_length"]
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, '*.wav'))
    
    print(f"Found {len(files)} wav files.")
    
    # Detection of device (Adding MPS support for Mac)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    for fpath in tqdm(files):
        fname = os.path.basename(fpath).replace('.wav', '.pt')
        save_path = os.path.join(output_dir, fname)
        
        # We REMOVE the skip logic here because we want to overwrite 
        # old 'tiny' features with new 'full' features.
        existing_f0 = None

        # Load
        audio, sr = torchaudio.load(fpath)
        # Mix to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample to 16k
        if sr != PREPROCESS_CONFIG["sample_rate"]:
            resampler = torchaudio.transforms.Resample(sr, PREPROCESS_CONFIG["sample_rate"])
            audio = resampler(audio)
            
        # Extract
        try:
            f0, loudness = extract_features(audio, PREPROCESS_CONFIG["sample_rate"], hop_length=hop_length, existing_f0=existing_f0)
            
            # Save
            num_frames = f0.shape[1]
            audio_target = audio[:, :num_frames * hop_length]
            
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
