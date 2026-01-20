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
#    - Why Frequency domain? Designing filters in frequency domain (magnitudes)
#      is strictly easier for a neural net than predicting time-domain taps.
#      We use Windowed-Sinc method (via iFFT) to convert H(f) -> h(t).
# ==============================================================================

class HarmonicSynthesizer(nn.Module):
    def __init__(self, n_harmonics: int = 100, sample_rate: int = 16000, hop_length: int = 160):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(self, f0: torch.Tensor, harmonic_amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Synthesize audio from pitch and harmonic amplitudes.

        Args:
            f0: [Batch, Time, 1]
            harmonic_amplitudes: [Batch, Time, n_harmonics]
        """
        # Upsample to audio rate
        # [B, T, C] -> [B, C, T] -> Interpolate -> [B, C, T*Hop] -> [B, T*Hop, C]
        target_len = f0.shape[1] * self.hop_length
        
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)
            
        f0_up = F.interpolate(f0.transpose(1, 2), size=target_len, mode='linear', align_corners=True).transpose(1, 2)
        amps_up = F.interpolate(harmonic_amplitudes.transpose(1, 2), size=target_len, mode='linear', align_corners=True).transpose(1, 2)
            
        harmonic_indices = torch.arange(1, self.n_harmonics + 1, device=f0.device).float() 
        frequencies = f0_up * harmonic_indices.unsqueeze(0).unsqueeze(0) # [B, T_audio, n_harmonics]

        # Integrate frequency to get phase
        phases = 2 * np.pi * torch.cumsum(frequencies / self.sample_rate, dim=1)
        
        sin_waves = torch.sin(phases) # [B, T_audio, n_harmonics]
        
        # Weighted sum
        harmonic_signals = sin_waves * amps_up # [B, T_audio, H]
        audio = torch.sum(harmonic_signals, dim=-1) # [B, T_audio]
        
        return audio


class FilteredNoiseSynthesizer(nn.Module):
    def __init__(self, n_bands: int = 200, window_size: int = None):
        """
        Synthesize noise by filtering white noise with time-varying filters.
        
        Args:
            n_bands (int): Number of frequency bands (usually window_size // 2 + 1 if linear).
            window_size (int): Size of the FIR filter window (impulse response length).
        """
        super().__init__()
        self.n_bands = n_bands
        if window_size is None:
            window_size = (n_bands - 1) * 2
        self.window_size = window_size
        self.register_buffer('hann_window', torch.hann_window(window_size))

    def forward(self, filter_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            filter_magnitudes (torch.Tensor): [Batch, Time, n_bands] 
                                             Magnitudes of the filter in frequency domain.
        Returns:
            torch.Tensor: Filtered noise audio [Batch, Time]
        """
        batch_size, n_frames, n_bands = filter_magnitudes.shape
        
        # 1. Generate White Noise
        # We need to match the time resolution. 
        # Usually 'filter_magnitudes' is at a lower frame rate (e.g. 100Hz) than audio (16kHz).
        # We assume the caller handles upsampling or we generate blocks.
        # For simplicity in this "Minimal" version, we assume the inputs are already 
        # upsampled to audio rate OR we process in blocks (LTV filter).
        # However, a proper LTV implementation is complex (overlap-add).
        # A simpler robust approximation used in standard DDSP:
        # Generate noise -> Apply STFT -> Multiply by Magnitudes -> iSTFT.
        
        # Let's assume filter_magnitudes is a frame-wise control signal.
        # We need to map this to an impulse response h(t).
        
        # Phase 0: Convert Magnitudes to Impulse Responses (Windowed Sinc mechanism essentially)
        # We treat magnitudes as the +ve frequency part of the spectrum.
        # We assume linear frequency scale for simplicity here.
        
        # Complex spectrum: Mag * e^(i*0) = Mag (Zero phase filters)
        # To get real impulse response, spectrum must be Hermitian symmetric.
        # But standard torch.irfft handles this if we just pass magnitude as complex with 0 phase?
        # Actually easier: 
        # impulse_responses = torch.fft.irfft(filter_magnitudes, n=self.window_size)
        # note: filter_magnitudes needs to be suitable size (window_size//2 + 1)
        
        # Shift 0-phase to center to make it causal/linear phase? 
        # Actually, for noise, random phase is inherent in the signal, so zero-phase filter is fine.
        # But we usually want to window it.
        
        # [B, T, n_bands] -> [B, T, window_size] (Impulse responses)
        impulse_responses = torch.fft.irfft(filter_magnitudes, n=self.window_size, dim=-1)
        
        # Window rotation to center (fftshift essentially)
        # irfft gives 0-delay filter (starts at t=0). We roll it to center?
        # Or Just apply Hann window to smooth edges.
        impulse_responses = impulse_responses * self.hann_window.view(1, 1, -1)

        # 2. Convolution (Time-Varying)
        # Since implementation of exact LTV convolution is slow in pure python,
        # we generate noise, block it, and convolve.
        # OR: We use the simpler "Filtered Noise" approach from the original DDSP paper:
        # "Noise is generated in the frequency domain... multiplied by the filter transfer function... then Inverse FFT"
        
        # Let's aim for the Frequency Domain generation which is O(T log T) and fast.
        
        # Generate random noise in frequency domain directly? 
        # Or: Generate White Noise w(t) -> STFT -> W(t, f)
        # Output = iSTFT( W(t, f) * H(t, f) )
        
        # We need target audio length. Assuming 1 hop per frame.
        # This implies we need to know the hop size relative to the input sampling.
        # FOR THIS SNIPPET: We will assume the input 'filter_magnitudes' aligns with STFT frames.
        
        param_frames = filter_magnitudes.shape[1]
        
        # Generate white noise matching the implicit duration
        # NOTE: This part requires external coordination of hop_size. 
        # For a "Module", we'll assume a standard hop (e.g. 160) or pass it in.
        # Let's fix a default hop for the "body" logic or allow it to be dynamic.
        # But usually synthesizers generate sample-by-sample or block-by-block.
        
        # Let's use the Impulse Response Convolution method which is robust to hop size mismatch,
        # but interpreted as: 
        # audio = noise * (filter_magnitudes interpreted as time-envelope per band)? No, that's subtractive.
        
        # Let's implement the specific "Frequency Sampling" method for clarity:
        # 1. Magnitudes [B, T, F] are interpreted as H(t, f).
        # 2. Uniform noise Uniform(-1, 1).
        # 3. Filter noise efficiently.
        
        # Implementation Detail:
        # We will use the torchaudio.functional.lfilter? No, that's IIR.
        # We will use simple frequency domain masking if standard DDSP does that.
        # Actually, original DDSP uses Time-varying FIR via Overlap-Add.
        
        # SIMPLIFIED ROBUST IMPLEMENTATION for this Portfolio:
        # Filtered Noise = Band-limited Noise summed up.
        # Almost equivalent to: H(t, f) * Noise(t, f)
        
        # We will do the "Filter in Frequency Domain" approach:
        # 1. Random Phase, Magnitude 1 (White Noise) in Frequency Domain? No.
        # 2. White Noise in time -> STFT.
        hop_length = 160 # Standard 16kHz/100fps
        n_fft = (self.n_bands - 1) * 2
        
        # Output length approx
        audio_length = param_frames * hop_length
        noise = torch.rand(batch_size, audio_length, device=filter_magnitudes.device) * 2 - 1
        
        noise_stft = torch.stft(noise, n_fft=n_fft, hop_length=hop_length, 
                                win_length=n_fft, window=torch.hann_window(n_fft, device=noise.device),
                                return_complex=True)
        
        # Resize filter_magnitudes to match STFT shape if needed, or assume they match 
        # (usually the decoder outputs exactly the control frames).
        # filter_magnitudes: [B, T, F] -> [B, F, T] for broadcasting with STFT [B, F, T]
        H = filter_magnitudes.transpose(1, 2)
        
        # Ensure shapes match (last frame truncation issues are common)
        # We crop to min length
        min_t = min(noise_stft.shape[2], H.shape[2])
        noise_stft = noise_stft[..., :min_t]
        H = H[..., :min_t]
        
        # Apply Filter
        filtered_stft = noise_stft * H # Treat H as real magnitude, phase 0 (or preserve noise phase)
        # Usage: we want to SHAPE the noise, so we multiply by Magnitude H. The Phase comes from the noise.
        # noise_stft is complex. H is real.
        filtered_stft = noise_stft * H # Broadcasting
        
        # Inverse STFT
        audio = torch.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length, 
                            win_length=n_fft, window=torch.hann_window(n_fft, device=noise.device))
        
        # Crop to target length (T * hop_length)
        target_len = param_frames * hop_length
        if audio.shape[-1] > target_len:
            audio = audio[..., :target_len]
        elif audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
            
        return audio
