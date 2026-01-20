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

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, FFT_sizes: List[int] = [512, 1024, 2048], 
                 hop_sizes: List[int] = [120, 240, 480], 
                 win_lengths: List[int] = [240, 600, 1200]):
        """
        Multi-Resolution Short-Time Fourier Transform Loss.
        
        Args:
            FFT_sizes (List[int]): List of FFT sizes for resolution bank.
            hop_sizes (List[int]): List of hop sizes corresponding to FFT sizes.
            win_lengths (List[int]): List of window lengths corresponding to FFT sizes.
        """
        super().__init__()
        assert len(FFT_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.loss_objs = nn.ModuleList()
        for fs, hs, wl in zip(FFT_sizes, hop_sizes, win_lengths):
            self.loss_objs.append(SingleResolutionSTFTLoss(fs, hs, wl))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the multi-resolution spectral loss.
        
        Args:
            x (torch.Tensor): Predicted audio [Batch, Time]
            y (torch.Tensor): Target audio [Batch, Time]
            
        Returns:
            torch.Tensor: Aggregated loss value.
        """
        total_loss = 0.0
        for loss_obj in self.loss_objs:
            total_loss += loss_obj(x, y)
            
        # We average by the number of resolutions to keep scale consistent
        return total_loss / len(self.loss_objs)


class SingleResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        # Define window as buffer so it moves to GPU with the model
        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate Spectral Convergence and Log-Magnitude Loss for a single resolution.
        """
        # 1. Compute STFTs
        # Note: We need to handle potential shape mismatches or padding if necessary,
        # but standard torchaudio.stft handles 1D signals well.
        # Inputs assumed to be [Batch, Time] or [Batch, 1, Time]. We squeeze channel if present.
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, 
                            self.window, return_complex=True)
        y_stft = torch.stft(y, self.fft_size, self.hop_size, self.win_length, 
                            self.window, return_complex=True)

        # 2. Compute Magnitudes
        # Create a small epsilon to prevent log(0)
        eps = 1e-7
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # 3. Spectral Convergence Loss
        # || |Y| - |X| ||_F / || |Y| ||_F
        # "How close is the overall energy distribution?"
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + eps)

        # 4. Log-Magnitude Loss
        # || log(|Y| + eps) - log(|X| + eps) ||_1
        # "How close are the details, perceptually weighted?"
        log_loss = F.l1_loss(torch.log(y_mag + eps), torch.log(x_mag + eps))

        # Total loss is a weighted sum (often 1.0 each in DDSP papers)
        return sc_loss + log_loss
