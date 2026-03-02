import torch
import torch.nn as nn
import torch.nn.functional as F
from synth import HarmonicSynthesizer, FilteredNoiseSynthesizer

# ==============================================================================
# Research Note: Neural Guitar Architecture (Brain & Body)
# ==============================================================================
# 1. The "Brain" (Decoder):
#    - We use a GRU (Recurrent Neural Network) because audio generation 
#      is fundamentally sequential and stateful (reverberation, decay).
#    - Input: Log-F0 and Log-Loudness (Perceptually relevant features).
#    - Inductive Bias: We don't ask the network to generate samples. 
#      We ask it to "perform" the instrument by controlling the knobs 
#      of the synthesizers.
#
# 2. Parameter Mapping:
#    - Harmonic Amplitudes: Force to sum-to-1 via Softmax (distribution).
#      Overall amplitude is controlled by the input Loudness capability.
#    - Noise Magnitudes: Sigmoid activation to bound filter response [0, 1].
# ==============================================================================

class NeuralGuitar(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        # Mandatory parameters from config - will raise KeyError if missing
        self.n_harmonics = config['n_harmonics']
        self.n_noise_bands = config['n_noise_bands']
        self.hidden_size = config['hidden_size']
        self.sample_rate = config['sample_rate']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # --- The Body (DSP) ---
        self.harmonic_synth = HarmonicSynthesizer(self.n_harmonics, self.sample_rate)
        self.noise_synth = FilteredNoiseSynthesizer(self.n_noise_bands)
        
        # --- The Brain (Decoder) ---
        # Input features: f0 (1) + loudness (1) = 2
        self.input_norm = nn.LayerNorm(2)
        self.gru = nn.GRU(
            input_size=2, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )
        self.mlp = nn.Linear(self.hidden_size, self.n_harmonics + self.n_noise_bands)
        
        # Stability: Small initial weights for the projection layer 
        # to prevent NaN explosions at start of training
        nn.init.normal_(self.mlp.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.mlp.bias, 0)
        


    def forward(self, f0: torch.Tensor, loudness: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0 (torch.Tensor): Fundamental frequency in Hz [Batch, Time, 1]
            loudness (torch.Tensor): Loudness signal (normalized dB usually) [Batch, Time, 1]
            
        Returns:
            torch.Tensor: Synthesized Audio [Batch, Time]
        """
        # 1. Feature Preprocessing
        # Log-scale f0 helps the network linearize pitch space
        # Guitar range is roughly 80Hz - 1000Hz. 
        # log(130) is ~4.8. Let's center it.
        log_f0 = (torch.log(f0 + 1e-7) - 4.8) / 2.0
        
        decoder_input = torch.cat([log_f0, loudness], dim=-1) # [B, T, 2]
        decoder_input = self.input_norm(decoder_input)
        
        # 2. Decoder (GRU)
        # x: [B, T, hidden_size]
        x, _ = self.gru(decoder_input)
        
        # 3. Parameter Projection
        # params: [B, T, H + N]
        params = self.mlp(x)
        
        # Split params
        harm_params = params[..., :self.n_harmonics]
        noise_params = params[..., self.n_harmonics:]
        
        # 4. Activation / Mapping
        # Harmonic Amps: 
        # We want a distribution that sums to 1, multiplied by a 'global' amplitude.
        # Here we model the distribution. The loudness envelope physically comes 
        # from the loudness input, but we usually let the network modulate it too.
        # Creating a "Amplitudes" tensor:
        
        # Softmax for distribution valid for timber
        harm_dist = F.softmax(harm_params, dim=-1)
        
        # Scale by input loudness (converted from log/dB to linear amp)? 
        # Or let the network predict absolute amplitude?
        # Standard DDSP: The network predicts the distribution, and we multiply 
        # by the original loudness feature (linearized) to enforce the volume contour.
        # Linear Average Loudness ~ 10^(loudness_db / 20)
        # For this minimal implementation, we assume 'loudness' is passed as 
        # linear amplitude envelope or we rely on the network to learn gain.
        # Let's simple use the Modified Softmax approach:
        # A = exp(harm_params) ...
        # But explicitly using input loudness as a control signal is stronger.
        
        # Setup: loudness is linear amplitude [0, 1]
        harm_amps = harm_dist * loudness 
        
        # Noise Params:
        # Magnitudes in [0, 1]
        noise_mags = torch.sigmoid(noise_params)
        
        # 5. Synthesis
        harmonic_audio = self.harmonic_synth(f0, harm_amps)
        noise_audio = self.noise_synth(noise_mags)
        
        # Sum
        final_audio = harmonic_audio + noise_audio
        
        return final_audio
