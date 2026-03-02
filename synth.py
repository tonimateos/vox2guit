import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# Research Note: Differentiable DSP Modules
# ==============================================================================
# 1. HarmonicSynthesizer (The Body - Strings): 
#    - Models the deterministic, periodic component.
#    - Uses additive synthesis of sinusoids at integer multiples of f0.
#    - Differentiable w.r.t parameters allows end-to-end training.
#
# 2. FilteredNoiseSynthesizer (The Body - Pick/Texture):
#    - Models the stochastic, non-periodic component (pick attack, fret noise).
#    - Method: Subtractive Synthesis. White noise -> Time-Varying FIR Filter.
# ==============================================================================

def resample_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    MPS-friendly linear interpolation for 1D signals.
    Args:
        x: [Batch, Channels, Time]
        target_len: target sequence length
    """
    if x.device.type == 'mps':
        # MPS workaround: Use 2D bilinear interpolation which is implemented
        return F.interpolate(x.unsqueeze(-1), size=(target_len, 1), mode='bilinear', align_corners=True).squeeze(-1)
    else:
        return F.interpolate(x, size=target_len, mode='linear', align_corners=True)

class HarmonicSynthesizer(nn.Module):
    def __init__(self, n_harmonics: int = 100, sample_rate: int = 16000, hop_length: int = 160):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(self, f0: torch.Tensor, harmonic_amplitudes: torch.Tensor) -> torch.Tensor:
        target_len = f0.shape[1] * self.hop_length
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
            
        f0_up = resample_1d(f0.transpose(1, 2), target_len).transpose(1, 2)
        amps_up = resample_1d(harmonic_amplitudes.transpose(1, 2), target_len).transpose(1, 2)
            
        harmonic_indices = torch.arange(1, self.n_harmonics + 1, device=f0.device).float() 
        frequencies = f0_up * harmonic_indices.unsqueeze(0).unsqueeze(0)

        phases = 2 * np.pi * torch.cumsum(frequencies / self.sample_rate, dim=1)
        sin_waves = torch.sin(phases)
        harmonic_signals = sin_waves * amps_up
        audio = torch.sum(harmonic_signals, dim=-1)
        
        return audio


class FilteredNoiseSynthesizer(nn.Module):
    def __init__(self, n_bands: int = 65):
        super().__init__()
        self.n_bands = n_bands

    def forward(self, filter_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            filter_magnitudes (torch.Tensor): [Batch, Time, n_bands] 
        """
        batch_size, n_frames, n_bands = filter_magnitudes.shape
        hop_length = 160
        n_fft = 1024 
        device = filter_magnitudes.device
        is_mps = device.type == 'mps'

        # 1. Interpolate filter_magnitudes to match STFT bins
        H_reshaped = filter_magnitudes.reshape(-1, 1, n_bands)
        H_interp = resample_1d(H_reshaped, n_fft // 2 + 1)
        H = H_interp.reshape(batch_size, n_frames, n_fft // 2 + 1).transpose(1, 2) # [B, F, T]

        # 2. Generate White Noise
        audio_length = n_frames * hop_length
        noise = torch.randn(batch_size, audio_length + n_fft, device=device)
        
        # 3. CPU Bridge for STFT/Filtering/ISTFT if on MPS
        # This avoids NotImplementedError and complex-multiplication crashes on MPS
        if is_mps:
            noise_cpu = noise.cpu()
            H_cpu = H.cpu()
            window_cpu = torch.hann_window(n_fft).cpu()
            
            # STFT on CPU
            noise_stft = torch.stft(
                noise_cpu, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=n_fft, 
                window=window_cpu,
                return_complex=True,
                center=True
            )
            
            min_t = min(noise_stft.shape[2], H_cpu.shape[2])
            noise_stft = noise_stft[..., :min_t]
            H_cpu = H_cpu[..., :min_t]
            
            # Multiply on CPU (safe)
            filtered_stft = noise_stft * H_cpu.abs()
            
            # ISTFT on CPU
            audio = torch.istft(
                filtered_stft, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=n_fft, 
                window=window_cpu,
                center=True
            ).to(device)
        else:
            # Standard path for CUDA/CPU
            window = torch.hann_window(n_fft, device=device)
            noise_stft = torch.stft(
                noise, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=n_fft, 
                window=window,
                return_complex=True,
                center=True
            ) 
            
            min_t = min(noise_stft.shape[2], H.shape[2])
            noise_stft = noise_stft[..., :min_t]
            H = H[..., :min_t]
            
            filtered_stft = noise_stft * H.abs()
            
            audio = torch.istft(
                filtered_stft, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=n_fft, 
                window=window,
                center=True
            )
        
        target_len = n_frames * hop_length
        if audio.shape[-1] > target_len:
            audio = audio[..., :target_len]
        elif audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
            
        return audio
