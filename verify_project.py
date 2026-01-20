import torch
import sys
import os

# Ensure we can import from local dir
sys.path.append(os.getcwd())

from model import NeuralGuitar
from loss import MultiResolutionSTFTLoss

def test_neural_guitar():
    print("========================================")
    print("Verifying Neural Guitar Project Components")
    print("========================================")
    
    # 1. Instantiate Model
    print("[1/3] Instantiating Model...")
    model = NeuralGuitar(n_harmonics=50, n_noise_bands=100, hidden_size=128)
    print("      Model created successfully.")
    
    # 2. Create Dummy Inputs
    print("[2/3] Generating Dummy Data...")
    batch_size = 2
    time_steps = 200 # Frames
    # f0: [B, T, 1] - random pitch between 200-400Hz
    f0 = torch.rand(batch_size, time_steps, 1) * 200 + 200 
    loudness = torch.rand(batch_size, time_steps, 1) # [0, 1]
    
    print(f"      Inputs: F0 {f0.shape}, Loudness {loudness.shape}")
    
    # 3. Forward Pass
    print("[3/3] Running Forward Pass...")
    try:
        audio = model(f0, loudness)
        print(f"      Output Audio Shape: {audio.shape}")
        
        # Verify Loss
        loss_fn = MultiResolutionSTFTLoss()
        # Create dummy target matching output
        target = torch.randn_like(audio)
        loss = loss_fn(audio, target)
        print(f"      Loss Computed: {loss.item()}")
        
        print("\n[SUCCESS] Project components verified OK.")
        
    except Exception as e:
        print(f"\n[FAILURE] Error during forward pass: {e}")
        raise e

if __name__ == "__main__":
    test_neural_guitar()
