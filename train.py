import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import NeuralGuitar
from data import NeuralGuitarDataset
from loss import MultiResolutionSTFTLoss

def train(args):
    # 1. Initialize W&B (Week 2 Alignment)
    wandb.init(project="vox2guit", config=args)
    config = wandb.config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data
    dataset = NeuralGuitarDataset(args.data_dir, sequence_length=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # 3. Model
    model = NeuralGuitar(
        n_harmonics=args.n_harmonics, 
        n_noise_bands=args.n_noise_bands, 
        hidden_size=args.hidden_size
    ).to(device)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MultiResolutionSTFTLoss().to(device)
    
    # 5. Resume logic
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint at epoch {start_epoch}")
    
    # 6. Training Loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            f0 = batch['f0'].to(device)
            loudness = batch['loudness'].to(device)
            target_audio = batch['audio'].to(device)
            
            # Forward
            # model(f0, loudness) triggers the forward() method
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
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        # Log Audio Sample periodically
        if (epoch + 1) % args.log_audio_every == 0:
            wandb.log({
                "source_f0": wandb.Histogram(f0.cpu().numpy()),
                "pred_audio": wandb.Audio(pred_audio[0].detach().cpu().numpy(), sample_rate=16000),
                "target_audio": wandb.Audio(target_audio[0].cpu().numpy(), sample_rate=16000)
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Preprocessed .pt files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq_len', type=int, default=16000)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--n_harmonics', type=int, default=101)
    parser.add_argument('--n_noise_bands', type=int, default=65)
    parser.add_argument('--log_audio_every', type=int, default=5)
    
    args = parser.parse_args()
    train(args)
