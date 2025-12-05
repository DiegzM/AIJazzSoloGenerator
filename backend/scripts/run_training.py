# ml/train.py

from logging import config
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / 'backend'
sys.path.insert(0, str(BACKEND_ROOT))

import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from functools import partial

from ml.models.transformer import JazzTransformer

# ============== Configuration ==============

class Config:
    # Paths
    processed_dir = PROJECT_ROOT / "data" / "processed"
    vocab_path = processed_dir / "vocab_v3.json"
    tokens_path = processed_dir / "token_pairs.pkl"
    checkpoint_dir = BACKEND_ROOT / "checkpoints"
    
    # Model
    d_model = 256
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    max_encoder_len = 700
    max_decoder_len = 4000
    
    # Training
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.01
    epochs = 100
    warmup_steps = 1000
    gradient_clip = 1.0
    
    # Data
    train_split = 0.9
    
    # Logging
    log_interval = 50
    save_interval = 5  # epochs
    use_wandb = False


# ============== Dataset ==============

class JazzDataset(Dataset):
    """Dataset for jazz solo generation."""
    
    def __init__(self, token_pairs, vocab, max_encoder_len=700, max_decoder_len=3500):
        self.data = token_pairs
        self.vocab = vocab
        self.pad_id = vocab["<PAD>"]
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle both tuple and dictionary formats
        if isinstance(item, dict):
            encoder_ids = item["encoder_input"]
            decoder_ids = item["decoder_target"]
        elif isinstance(item, (list, tuple)):
            encoder_ids = item[0]
            decoder_ids = item[1]
        else:
            raise ValueError(f"Unexpected item type: {type(item)}")
        
        # Data is ALREADY integer IDs - just convert to tensors
        return {
            "encoder_input": torch.tensor(encoder_ids, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_ids[:-1], dtype=torch.long),
            "decoder_target": torch.tensor(decoder_ids[1:], dtype=torch.long),
        }

def collate_fn(batch, pad_id=0):
    """Collate function with padding."""
    
    # Find max lengths in batch
    max_enc_len = max(item["encoder_input"].size(0) for item in batch)
    max_dec_len = max(item["decoder_input"].size(0) for item in batch)
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    
    for item in batch:
        # Pad encoder
        enc_pad = max_enc_len - item["encoder_input"].size(0)
        encoder_inputs.append(
            torch.cat([item["encoder_input"], torch.full((enc_pad,), pad_id, dtype=torch.long)])
        )
        
        # Pad decoder input
        dec_pad = max_dec_len - item["decoder_input"].size(0)
        decoder_inputs.append(
            torch.cat([item["decoder_input"], torch.full((dec_pad,), pad_id, dtype=torch.long)])
        )
        
        # Pad decoder target
        decoder_targets.append(
            torch.cat([item["decoder_target"], torch.full((dec_pad,), pad_id, dtype=torch.long)])
        )
    
    return {
        "encoder_input": torch.stack(encoder_inputs),
        "decoder_input": torch.stack(decoder_inputs),
        "decoder_target": torch.stack(decoder_targets),
    }


# ============== Training ==============

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, config, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        decoder_target = batch["decoder_target"].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(encoder_input, decoder_input)
        
        # Compute loss (flatten for cross entropy)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1)
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        print(f"Training loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)
            
            logits = model(encoder_input, decoder_input)
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint["epoch"], checkpoint["loss"]


# ============== Main ==============

def main():
    config = Config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(config.vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_id = vocab["<PAD>"]
    print(f"Vocabulary size: {vocab_size}")
    
    # Load token pairs
    print("Loading token pairs...")
    with open(config.tokens_path, "rb") as f:
        token_pairs_raw = pickle.load(f)

    # Filter out problematic samples
    token_pairs = []
    for item in token_pairs_raw:
        if isinstance(item, dict):
            enc = item["encoder_input"]
            dec = item["decoder_target"]
        else:
            enc, dec = item[0], item[1]
        
        # Keep only samples with sufficient length
        if len(enc) >= 5 and len(dec) >= 5:
            token_pairs.append(item)

    print(f"Filtered samples: {len(token_pairs_raw)} -> {len(token_pairs)}")
    
    # Split data
    split_idx = int(len(token_pairs) * config.train_split)
    train_data = token_pairs[:split_idx]
    val_data = token_pairs[split_idx:]
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = JazzDataset(train_data, vocab, config.max_encoder_len, config.max_decoder_len)
    val_dataset = JazzDataset(val_data, vocab, config.max_encoder_len, config.max_decoder_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        num_workers=0,  # Use 0 for macOS to avoid multiprocessing issues
        pin_memory=False  # MPS doesn't support pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("Creating model...")
    model = JazzTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_encoder_len=config.max_encoder_len,
        max_decoder_len=config.max_decoder_len,
        pad_id=pad_id
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 10,  # Restart every 10 epochs
        T_mult=2
    )
    
    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.05)

    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, config, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                config.checkpoint_dir / "best_model.pt"
            )
            print(f"  New best model saved! (val_loss = {val_loss:.4f})")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, config.epochs, val_loss,
        config.checkpoint_dir / "final_model.pt"
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()