# train.py - 학습 루프 및 평가

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import os

from model import TextTo3DModel, chamfer_distance

class PointCloudDataset(Dataset):
    """
    Point Cloud Dataset from preprocessed NPZ files
    """
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        
        # Load index
        with open(self.data_dir / "index.json", 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        
        # Train/Val split (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.data_list))
        split_idx = int(len(self.data_list) * 0.8)
        
        if split == 'train':
            self.data_list = [self.data_list[i] for i in indices[:split_idx]]
        else:
            self.data_list = [self.data_list[i] for i in indices[split_idx:]]
        
        print(f"{split} dataset: {len(self.data_list)} samples")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Load NPZ file
        data_path = self.data_list[idx]['file']
        data = np.load(data_path)
        
        return {
            'points': torch.from_numpy(data['points']).float(),  # (1024, 3)
            'description': str(data['description'])
        }

def train_epoch(model, dataloader, optimizer, device):
    """
    단일 epoch 학습
    """
    model.train()
    model.decoder.train()  # Decoder만 학습 모드
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        points = batch['points'].to(device)  # (batch_size, 1024, 3)
        descriptions = batch['description']  # list of strings
        
        # Forward pass
        pred_points = model(descriptions)  # (batch_size, 1024, 3)
        
        # Compute loss
        loss = chamfer_distance(pred_points, points)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, dataloader, device):
    """
    Validation
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            points = batch['points'].to(device)
            descriptions = batch['description']
            
            # Forward pass
            pred_points = model(descriptions)
            
            # Compute loss
            loss = chamfer_distance(pred_points, points)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train(
    data_dir='./data/processed',
    num_epochs=500,
    batch_size=4,
    learning_rate=1e-4,
    num_points=1024,
    latent_dim=128,
    checkpoint_dir='./checkpoints',
    log_dir='./logs',
    early_stopping_patience=50
):
    """
    전체 학습 파이프라인
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Dataset & DataLoader
    train_dataset = PointCloudDataset(data_dir, split='train')
    val_dataset = PointCloudDataset(data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Model
    model = TextTo3DModel(
        num_points=num_points,
        latent_dim=latent_dim,
        device=device
    )
    
    # Optimizer (only decoder parameters)
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Scheduler step
        scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
            print(f"✓ Best model saved (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    writer.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    # 학습 실행
    train(
        data_dir='./data/processed',
        num_epochs=500,
        batch_size=4,
        learning_rate=1e-4,
        num_points=1024,
        latent_dim=128,
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        early_stopping_patience=50
    )