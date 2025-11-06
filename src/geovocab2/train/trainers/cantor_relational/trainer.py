# geovocab2/train/trainers/cantor_relational_trainer.py

"""
Trainer for Cantor Relational Model - Colab Optimized

Features:
- Automatic CLIP/T5 embedding generation
- VAE-style loss tracking
- Checkpoint saving/loading
- Weights & Biases integration
- Progress visualization
- Memory efficient batching
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from tqdm.auto import tqdm
import wandb
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

from geovocab2.train.model.relational.cantor_relational import create_cantor_relational, CantorRelationalConfig
from geovocab2.train.losses.cantor import create_vae_loss


@dataclass
class TrainerConfig:
    """Training configuration."""
    # Model
    model_dim: int = 512
    num_heads: int = 8
    num_blocks: int = 6
    seq_len: int = 77
    cantor_depth: int = 8
    local_window: int = 64

    # Loss
    beta_kl: float = 0.1
    beta_cross: float = 0.05
    beta_sparse: float = 0.001
    recon_type: str = 'mse'  # 'mse' or 'cosine'

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine' or 'linear'

    # Data
    dataset_path: Optional[str] = None  # Path to text file or dataset
    max_samples: Optional[int] = None  # Limit dataset size

    # Checkpointing
    checkpoint_dir: str = '/content/checkpoints'
    save_every: int = 1000  # Save every N steps
    keep_last_n: int = 3  # Keep only last N checkpoints

    # Logging
    use_wandb: bool = True
    wandb_project: str = 'cantor-relational'
    wandb_entity: Optional[str] = None
    log_every: int = 50

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True  # Use AMP for faster training

    # Misc
    seed: int = 42
    num_workers: int = 2


class TextEmbeddingDataset(Dataset):
    """
    Dataset that generates CLIP and T5 embeddings on-the-fly.
    """

    def __init__(
            self,
            texts: List[str],
            clip_tokenizer: CLIPTokenizer,
            clip_model: CLIPTextModel,
            t5_tokenizer: T5Tokenizer,
            t5_model: T5EncoderModel,
            device: str = 'cuda',
            max_length: int = 77
    ):
        self.texts = texts
        self.clip_tokenizer = clip_tokenizer
        self.clip_model = clip_model
        self.t5_tokenizer = t5_tokenizer
        self.t5_model = t5_model
        self.device = device
        self.max_length = max_length

        # Move models to device and eval mode
        self.clip_model.to(device).eval()
        self.t5_model.to(device).eval()

    def __len__(self):
        return len(self.texts)

    @torch.no_grad()
    def __getitem__(self, idx):
        text = self.texts[idx]

        # CLIP embedding
        clip_tokens = self.clip_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_output = self.clip_model(**clip_tokens)
        clip_embed = clip_output.last_hidden_state.squeeze(0)  # (77, 512)

        # T5 embedding
        t5_tokens = self.t5_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        t5_output = self.t5_model(**t5_tokens)
        t5_embed = t5_output.last_hidden_state.squeeze(0)  # (77, 512)

        return {
            'clip': clip_embed.cpu(),  # Move to CPU for batching
            't5': t5_embed.cpu(),
            'text': text
        }


class CantorRelationalTrainer:
    """
    Trainer for Cantor Relational Model.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Initialize models
        print("Initializing models...")
        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize scheduler
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                name=f"cantor_relational_{config.num_blocks}blocks"
            )

    def _build_model(self) -> nn.Module:
        """Build Cantor Relational model."""
        model = create_cantor_relational(
            dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_blocks=self.config.num_blocks,
            seq_len=self.config.seq_len,
            cantor_depth=self.config.cantor_depth,
            local_window=self.config.local_window
        )

        model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")

        return model

    def _build_loss_fn(self):
        """Build VAE loss function."""
        return create_vae_loss(
            beta_kl=self.config.beta_kl,
            beta_cross=self.config.beta_cross,
            beta_sparse=self.config.beta_sparse,
            recon_type=self.config.recon_type
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def prepare_data(self, texts: List[str]) -> DataLoader:
        """
        Prepare dataset and dataloader.

        Args:
            texts: List of text strings

        Returns:
            DataLoader
        """
        print("Loading CLIP and T5 models...")

        # Load CLIP
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # Load T5
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5EncoderModel.from_pretrained("t5-small")

        # Limit dataset if specified
        if self.config.max_samples:
            texts = texts[:self.config.max_samples]

        print(f"Dataset size: {len(texts)} samples")

        # Create dataset
        dataset = TextEmbeddingDataset(
            texts=texts,
            clip_tokenizer=clip_tokenizer,
            clip_model=clip_model,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            device=self.device,
            max_length=self.config.seq_len
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        return dataloader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch dict with 'clip' and 't5' embeddings

        Returns:
            Dict of metrics
        """
        clip_embed = batch['clip'].to(self.device)  # (batch, 77, 512)
        t5_embed = batch['t5'].to(self.device)  # (batch, 77, 512)

        # Mixed precision training
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Forward pass
                clip_out, t5_out = self.model(clip_embed, t5_embed, return_both=True)

                # Compute loss
                loss, components = self.loss_fn(
                    clip_in=clip_embed,
                    clip_out=clip_out,
                    t5_in=t5_embed,
                    t5_out=t5_out,
                    return_components=True
                )

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            clip_out, t5_out = self.model(clip_embed, t5_embed, return_both=True)

            loss, components = self.loss_fn(
                clip_in=clip_embed,
                clip_out=clip_out,
                t5_in=t5_embed,
                t5_out=t5_out,
                return_components=True
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

        # Convert to float for logging
        metrics = {k: v.item() for k, v in components.items()}

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Dict of average metrics
        """
        self.model.train()
        epoch_metrics = {}

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)

            # Update global step
            self.global_step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'recon': f"{metrics['reconstruction']:.4f}",
                'kl': f"{metrics['kl_divergence']:.4f}"
            })

            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_every == 0:
                wandb.log({
                    **{f'train/{k}': v for k, v in metrics.items()},
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.epoch,
                    'train/step': self.global_step
                })

            # Save checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

        return avg_metrics

    def train(self, dataloader: DataLoader):
        """
        Full training loop.

        Args:
            dataloader: Training dataloader
        """
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total steps per epoch: {len(dataloader)}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train epoch
            metrics = self.train_epoch(dataloader)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {metrics['total']:.4f}")
            print(f"  Reconstruction: {metrics['reconstruction']:.4f}")
            print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")
            print(f"  Cross-Modal: {metrics['cross_modal']:.4f}")

            # Save best model
            if metrics['total'] < self.best_loss:
                self.best_loss = metrics['total']
                self.save_checkpoint(best=True)
                print(f"  ✓ New best model saved! Loss: {self.best_loss:.4f}")

        print("\n✓ Training complete!")

        if self.config.use_wandb:
            wandb.finish()

    def save_checkpoint(self, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if best:
            path = Path(self.config.checkpoint_dir) / 'best_model.pt'
        else:
            path = Path(self.config.checkpoint_dir) / f'checkpoint_step_{self.global_step}.pt'

        torch.save(checkpoint, path)

        # Clean old checkpoints
        if not best:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob('checkpoint_step_*.pt')],
            key=lambda x: int(x.stem.split('_')[-1])
        )

        # Remove old checkpoints
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        print(f"Loaded checkpoint from step {self.global_step}")


def load_text_dataset(path: str) -> List[str]:
    """
    Load text dataset from file.

    Supports:
    - .txt files (one prompt per line)
    - .json files (list of strings or dict with 'text' key)

    Args:
        path: Path to dataset file

    Returns:
        List of text strings
    """
    path = Path(path)

    if path.suffix == '.txt':
        with open(path) as f:
            texts = [line.strip() for line in f if line.strip()]

    elif path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = data
            elif isinstance(data, dict) and 'text' in data:
                texts = data['text']
            else:
                raise ValueError("JSON must be list of strings or dict with 'text' key")

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print(f"Loaded {len(texts)} texts from {path}")
    return texts