import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ==============================================================================
# Research Note: Multi-Resolution STFT Loss
# ==============================================================================
# Why specialize this?
# 1. Audio perception is non-linear and operates on multiple time-frequency scales.
#    A single FFT size forces a trade-off: 
#    - Large FFT = good frequency resolution (pitch), bad time resolution (transients/attacks).
#    - Small FFT = good time resolution, bad frequency resolution.
# 2. To capture the full fidelity of a "Neural Guitar" (which has sharp attacks AND sustained harmonics),
#    we MUST evaluate the loss across a bank of FFT sizes simultaneously.
# 3. We use the L1 distance on log-magnitudes to model the logarithmic nature of human hearing (Weber-Fechner law).
# ==============================================================================

def safe_stft_mag(x, n_fft, hop_length, win_length, window):
    """
    STFT that moves to CPU if on MPS, calculates magnitude, and returns to original device.
    This avoids complex-tensor 'NotImplementedError' issues on MPS.
    """
    if x.device.type == 'mps':
        device = x.device
        x_cpu = x.cpu()
        window_cpu = window.cpu()
        stft_cpu = torch.stft(x_cpu, n_fft, hop_length, win_length, window_cpu, center=True, return_complex=True)
        mag_cpu = torch.abs(stft_cpu)
        return mag_cpu.to(device)
    else:
        stft = torch.stft(x, n_fft, hop_length, win_length, window, center=True, return_complex=True)
        return torch.abs(stft)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, FFT_sizes: List[int], hop_sizes: List[int], win_lengths: List[int], mag_loss_weight: float = 1.0):
        """
        Multi-Resolution Short-Time Fourier Transform Loss.
        
        Args:
            FFT_sizes (List[int]): List of FFT sizes for resolution bank.
            hop_sizes (List[int]): List of hop sizes corresponding to FFT sizes.
            win_lengths (List[int]): List of window lengths corresponding to FFT sizes.
            mag_loss_weight (float): Multiplier for the log-magnitude loss.
        """
        super().__init__()
        assert len(FFT_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.loss_objs = nn.ModuleList()
        for fs, hs, wl in zip(FFT_sizes, hop_sizes, win_lengths):
            self.loss_objs.append(SingleResolutionSTFTLoss(fs, hs, wl, mag_loss_weight))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the multi-resolution spectral loss.
        """
        total_loss = 0.0
        for loss_obj in self.loss_objs:
            total_loss += loss_obj(x, y)
            
        return total_loss / len(self.loss_objs)


class SingleResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_length: int, mag_loss_weight: float = 1.0):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.mag_loss_weight = mag_loss_weight
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate Spectral Convergence and Log-Magnitude Loss for a single resolution.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        x_mag = safe_stft_mag(x, self.fft_size, self.hop_size, self.win_length, self.window)
        y_mag = safe_stft_mag(y, self.fft_size, self.hop_size, self.win_length, self.window)

        eps = 1e-7

        # 1. Spectral Convergence Loss
        y_norm = torch.norm(y_mag, p="fro")
        diff_norm = torch.norm(y_mag - x_mag, p="fro")
        
        if torch.isnan(y_norm) or torch.isinf(y_norm) or torch.isnan(diff_norm) or torch.isinf(diff_norm):
            print(f"!!! NaN/Inf detected in Spectral Convergence calculation (FFT: {self.fft_size})")
            raise RuntimeError("Stop training: NaN/Inf in spectral convergence")
            
        sc_loss = diff_norm / (y_norm + eps)

        # 2. Log-Magnitude Loss (Weighed by config)
        log_y = torch.log(y_mag + eps)
        log_x = torch.log(x_mag + eps)
        
        if torch.isnan(log_y).any() or torch.isinf(log_y).any() or torch.isnan(log_x).any() or torch.isinf(log_x).any():
            print(f"!!! NaN/Inf detected in Log-Magnitude calculation (FFT: {self.fft_size})")
            raise RuntimeError("Stop training: NaN/Inf in log-magnitude loss")
            
        log_loss = F.l1_loss(log_y, log_x)

        return sc_loss + self.mag_loss_weight * log_loss
