import os
import torch
import numpy as np
import librosa
from scipy.io import wavfile
import json
from core import NeuralGuitarCore

def run_test():
    print("--- Running End-to-End Regression Test ---")
    
    # 1. Initialize Core (Seeding for determinism inside core isn't there, so we do it here)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    core = NeuralGuitarCore(
        checkpoint_path="checkpoints/latest.pth",
        config_path="config.json",
        config_name="tiny"
    )
    
    # 2. Paths
    ref_input = "tests/reference_input.wav"
    golden_ref = "tests/golden_output.wav"
    test_output = "tests/current_test_output.wav"
    
    if not os.path.exists(ref_input):
        print(f"Error: Missing {ref_input}")
        return False

    # 3. Run Pipeline
    print(f"Processing {ref_input}...")
    audio_orig, audio_resynth, f0, loudness = core.process_audio(ref_input)
    
    # 3. Save current output for reference/debugging
    wavfile.write(test_output, core.config["sample_rate"], audio_resynth)
    
    # 4. Golden Reference Logic
    if not os.path.exists(golden_ref):
        print(f"First run detected! Creating golden reference at {golden_ref}")
        wavfile.write(golden_ref, core.config["sample_rate"], audio_resynth)
        print("Done. Run the test again to verify consistency.")
        return True

    # 5. Compare
    print("Comparing current output with golden reference...")
    audio_gold, _ = librosa.load(golden_ref, sr=16000)
    
    # audio_resynth is already at 16k
    audio_test = audio_resynth
    
    # Check length
    if len(audio_test) != len(audio_gold):
        print(f"FAILED: Length mismatch. Current: {len(audio_test)}, Gold: {len(audio_gold)}")
        return False
    
    # Check values (Mean Squared Error)
    mse = np.mean((audio_test - audio_gold)**2)
    print(f"Mean Squared Error: {mse:.8e}")
    
    # High precision required for deterministic neural net outputs on same CPU
    # We allow a tiny tolerance for floating point non-determinism if any
    if mse < 1e-10:
        print("✅ SUCCESS: Output matches reference!")
        return True
    else:
        print("❌ FAILED: Output deviates from reference!")
        return False

if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
