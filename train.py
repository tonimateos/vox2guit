import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import json

from model import NeuralGuitar
from data import NeuralGuitarDataset
from loss import MultiResolutionSTFTLoss

def train(args):
    # Load external config
    with open(args.config_file, "r") as f:
        all_configs = json.load(f)
    net_config = all_configs[args.config_name]
    
    # 1. Initialize W&B (Week 2 Alignment)
    wandb.init(project="vox2guit", config=args)
    config = wandb.config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data
    dataset = NeuralGuitarDataset(args.data_dir, sequence_length=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # 3. Model
    model = NeuralGuitar(config=net_config).to(device)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MultiResolutionSTFTLoss().to(device)
    
    # 5. Resume logic
    start_epoch = 0
    checkpoint_to_load = args.resume
    
    # Auto-resume from latest.pth if no path provided
    if checkpoint_to_load is None:
        auto_latest = os.path.join(args.checkpoint_dir, "latest.pth")
        if os.path.exists(auto_latest):
            checkpoint_to_load = auto_latest
            print(f"Auto-resuming from latest checkpoint: {checkpoint_to_load}")

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"Loading checkpoint: {checkpoint_to_load}")
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        
        # Robust loading for both full dict and legacy state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            # Legacy format (just weights)
            model.load_state_dict(checkpoint)
            
        print(f"Success! Resuming from epoch {start_epoch}")
    
    # 6. Training Loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch_idx, batch in enumerate(pbar):
                f0 = batch['f0'].to(device)
                loudness = batch['loudness'].to(device)
                target_audio = batch['audio'].to(device)
                
                # Forward
                pred_audio = model(f0, loudness)
                
                # Loss
                loss = loss_fn(pred_audio, target_audio)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log to W&B
                if batch_idx % 10 == 0:
                    wandb.log({"train_loss": loss.item()})
                
                pbar.set_postfix({"loss": loss.item()})
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            
            # Save Checkpoint
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            latest_path = os.path.join(args.checkpoint_dir, "latest.pth")
            
            torch.save(checkpoint_data, checkpoint_path)
            torch.save(checkpoint_data, latest_path) # Always keep a latest.pth for easy resume
            
            # Log Audio Sample periodically
            if (epoch + 1) % args.log_audio_every == 0:
                # Normalize for W&B listening
                audio_to_log = pred_audio[0].detach().cpu().numpy()
                audio_to_log = audio_to_log / (np.max(np.abs(audio_to_log)) + 1e-7)
                
                wandb.log({
                    "source_f0": wandb.Histogram(f0.cpu().numpy()),
                    "pred_audio": wandb.Audio(audio_to_log, sample_rate=16000),
                    "target_audio": wandb.Audio(target_audio[0].cpu().numpy(), sample_rate=16000)
                })
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving state...")
    finally:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Preprocessed .pt files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq_len', type=int, default=16000)
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--config_name', type=str, default='tiny')
    parser.add_argument('--log_audio_every', type=int, default=5)
    
    args = parser.parse_args()
    train(args)
