import torch
import os
import argparse

def shrink_checkpoint(input_path, output_path):
    print(f"--- Shrinking Checkpoint: {input_path} ---")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load full checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Check if it's already a bare state dict
    if 'model_state_dict' in checkpoint:
        print("Found full training checkpoint (weights + optimizer).")
        shrunk_state_dict = checkpoint['model_state_dict']
    else:
        print("File is already a bare state dict (weights only). No further shrinking possible.")
        shrunk_state_dict = checkpoint

    # Save only the model weights
    torch.save(shrunk_state_dict, output_path)
    
    # Calculate savings
    old_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Done!")
    print(f"  Old Size: {old_size:.2f} MB")
    print(f"  New Size: {new_size:.2f} MB")
    print(f"  Saved:    {old_size - new_size:.2f} MB ({(1 - new_size/old_size)*100:.1f}%)")
    print(f"--- Shrunk checkpoint saved to: {output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shrink Neural Guitar checkpoints for deployment.")
    parser.add_argument("--input", type=str, default="checkpoints/latest.pth", help="Path to full checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/latest_shrunk.pth", help="Path to save shrunk weights")
    
    args = parser.parse_args()
    shrink_checkpoint(args.input, args.output)
