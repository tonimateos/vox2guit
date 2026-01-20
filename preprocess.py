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

def extract_features(audio: torch.Tensor, sample_rate: int, hop_length: int = 160):
    """
    Extract f0 and loudness from audio.
    
    Args:
        audio (torch.Tensor): Audio samples [1, Time]
        sample_rate (int): Sampling rate (e.g. 16000)
    
    Returns:
        f0, loudness
    """
    # 1. Extract Loudness (A-Weighted logic is complex, using RMS for Minimal MVP)
    # We window the signal to calculate envelope
    # Approximate A-weighting? Torchaudio has transforms, but simple RMS 
    # on STFT or windowed frames is standard for "Loudness".
    
    # Calculate RMS in windows matching the hop_length to get frame-wise loudness
    # Unfold creates [N_frames, Window_Size]
    # We use a window size e.g. 1024 centered?
    # Simpler: standard torchaudio MelSpectrogram or just framing.
    
    # Let's use simple frame-wise RMS.
    frame_length = 1024
    # Pad to center
    audio_pad = torch.nn.functional.pad(audio, (frame_length // 2, frame_length // 2))
    audio_frames = audio_pad.unfold(1, frame_length, hop_length) # [1, Frames, Win]
    
    # RMS = sqrt(mean(x**2))
    loudness = torch.sqrt(torch.mean(audio_frames**2, dim=-1)) # [1, Frames]
    
    # Convert to dB? standard DDSP usually keeps linear or log.
    # We'll return linear, let Model handle log.
    loudness = loudness.unsqueeze(-1) # [1, Frames, 1]
    
    # 2. Extract Pitch with CREPE
    # CREPE expects audio on specific device.
    # Note: Torchcrepe handles chunking automatically for long files, 
    # but we should ensure device is correct.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CREPE requires 16k usually.
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(audio.device)
        audio_16k = resampler(audio)
    else:
        audio_16k = audio

    # F0 extraction
    # output: [Batch, Time] (Time matches hop_length logic relative to CREPE's 10ms default?)
    # CREPE default hop is 10ms (160 samples at 16k).
    # ensure our hop_length matches what we want. 
    # If we want custom hop (e.g. 160), torchcrepe supports it.
    
    f0, confidence = torchcrepe.predict(
        audio_16k, 
        sample_rate=16000, 
        hop_length=hop_length, 
        fmin=50, 
        fmax=2000, 
        model='tiny', 
        batch_size=2048,
        device=device,
        return_periodicity=True
    )
    
    # Filter f0 by confidence? 
    # For training, we usually keep it, or mask silence.
    # We return clean f0.
    f0 = f0.unsqueeze(-1) # [Batch, Frames, 1]
    
    # Match lengths (Loudness vs CREPE might differ by 1-2 frames due to padding)
    min_len = min(f0.shape[1], loudness.shape[1])
    f0 = f0[:, :min_len, :]
    loudness = loudness[:, :min_len, :]
    
    return f0, loudness


def preprocess_dataset(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, '*.wav'))
    
    print(f"Found {len(files)} wav files.")
    
    import warnings
    warnings.filterwarnings("ignore") # Ignore some torchaudio warnings

    for fpath in tqdm(files):
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
            f0, loudness = extract_features(audio, 16000, hop_length=160)
            
            # Save
            # We save triplets: inputs(f0, loudness) -> target(audio)
            # We need to crop audio to match features (striding)
            # Audio len = Frames * hop_length
            num_frames = f0.shape[1]
            audio_target = audio[:, :num_frames * 160]
            
            fname = os.path.basename(fpath).replace('.wav', '.pt')
            save_path = os.path.join(output_dir, fname)
            
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
    args = parser.parse_args()
    
    preprocess_dataset(args.input_dir, args.output_dir)
