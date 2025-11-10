# geovocab2/train/trainers/vae_lyra/trainer.py

"""
Trainer for VAE Lyra - Multi-Modal Variational Autoencoder
CLIP-L + CLIP-G + T5-XXL for SDXL Compatibility

Install via:
    !pip install git+https://github.com/AbstractPhil/geovocab2.git

Usage:
    from geovocab2.train.trainers.vae_lyra_trainer import (
        VAELyraTrainer, VAELyraTrainerConfig
    )
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    T5Tokenizer
)
from huggingface_hub import HfApi, hf_hub_download, create_repo, upload_file
from tqdm.auto import tqdm
import wandb
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import requests
import random
from collections import Counter
import shutil

from geovocab2.train.model.vae.vae_lyra import (
    MultiModalVAE,
    MultiModalVAEConfig,
    MultiModalVAELoss,
    SingleModalityVAE,
    FusionStrategy
)
from geovocab2.data.prompt.symbolic_tree import SynthesisSystem


@dataclass
class VAELyraTrainerConfig:
    """Training configuration for VAE Lyra with SDXL support."""

    # Model architecture - SDXL three-modality setup
    modality_dims: Dict[str, int] = None  # {"clip_l": 768, "clip_g": 1280, "t5_xl": 2048}
    latent_dim: int = 768
    seq_len: int = 77
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.1

    # Fusion
    fusion_strategy: str = "cantor"
    fusion_heads: int = 8
    fusion_dropout: float = 0.1

    # Loss weights
    beta_kl: float = 0.1
    beta_reconstruction: float = 1.0
    beta_cross_modal: float = 0.05
    recon_type: str = 'mse'

    # NEW: Per-modality reconstruction weights
    modality_recon_weights: Dict[str, float] = None  # <-- ADD THIS
    cross_modal_projection_dim: int = 768  # <-- ADD THIS TOO

    # KL annealing
    use_kl_annealing: bool = True
    kl_anneal_epochs: int = 10
    kl_start_beta: float = 0.0

    # Training hyperparameters
    batch_size: int = 16  # Reduced for larger models
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine' or 'onecycle'

    # Data
    num_samples: int = 10000
    synthetic_ratio: float = 0.15

    # Checkpointing
    checkpoint_dir: str = './checkpoints_lyra_sdxl'
    save_every: int = 1000
    keep_last_n: int = 3

    # HuggingFace Hub
    hf_repo: str = "AbstractPhil/vae-lyra-sdxl"
    push_to_hub: bool = True
    push_every: int = 2000
    auto_load_from_hub: bool = True

    # Logging
    use_wandb: bool = True
    wandb_project: str = 'vae-lyra-sdxl'
    wandb_entity: Optional[str] = None
    log_every: int = 50

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True

    # Misc
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        if self.modality_dims is None:
            self.modality_dims = {
                "clip_l": 768,
                "clip_g": 1280,
                "t5_xl": 2048
            }

        # Default weights: prioritize CLIP outputs for SDXL
        if self.modality_recon_weights is None:
            self.modality_recon_weights = {
                "clip_l": 1.0,  # Full weight - SDXL primary
                "clip_g": 1.0,  # Full weight - SDXL primary
                "t5_xl": 0.3  # Lower weight - auxiliary signal
            }


class TextEmbeddingDataset(Dataset):
    """Dataset that generates CLIP-L, CLIP-G, and T5-XXL embeddings on-the-fly."""

    def __init__(
            self,
            texts: List[str],
            clip_l_tokenizer: CLIPTokenizer,
            clip_l_model: CLIPTextModel,
            clip_g_tokenizer: CLIPTokenizer,
            clip_g_model: CLIPTextModelWithProjection,
            t5_tokenizer: T5Tokenizer,
            t5_model: T5EncoderModel,
            device: str = 'cuda',
            max_length: int = 77
    ):
        self.texts = texts
        self.clip_l_tokenizer = clip_l_tokenizer
        self.clip_l_model = clip_l_model
        self.clip_g_tokenizer = clip_g_tokenizer
        self.clip_g_model = clip_g_model
        self.t5_tokenizer = t5_tokenizer
        self.t5_model = t5_model
        self.device = device
        self.max_length = max_length

        # Move models to device and set to eval
        self.clip_l_model.to(device).eval()
        self.clip_g_model.to(device).eval()
        self.t5_model.to(device).eval()

    def __len__(self):
        return len(self.texts)

    @torch.no_grad()
    def __getitem__(self, idx):
        text = self.texts[idx]

        # CLIP-L embedding (SD1.5 / SDXL text_encoder)
        clip_l_tokens = self.clip_l_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_l_output = self.clip_l_model(**clip_l_tokens)
        clip_l_embed = clip_l_output.last_hidden_state.squeeze(0)

        # CLIP-G embedding (SDXL text_encoder_2)
        clip_g_tokens = self.clip_g_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_g_output = self.clip_g_model(**clip_g_tokens)
        clip_g_embed = clip_g_output.last_hidden_state.squeeze(0)

        # T5-XXL embedding
        t5_tokens = self.t5_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        t5_output = self.t5_model(**t5_tokens)
        t5_embed = t5_output.last_hidden_state.squeeze(0)

        return {
            'clip_l': clip_l_embed.cpu(),
            'clip_g': clip_g_embed.cpu(),
            't5_xl': t5_embed.cpu(),
            'text': text
        }


class VAELyraTrainer:
    """Trainer for VAE Lyra - SDXL-compatible Multi-Modal VAE."""

    def __init__(self, config: VAELyraTrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # HuggingFace API
        self.hf_api = HfApi()
        self._init_hf_repo()

        print("🎵 Initializing VAE Lyra (SDXL Edition)...")
        print(f"   Modalities: CLIP-L (768), CLIP-G (1280), T5-XXL (2048)")

        # Try to load from HF first
        if config.auto_load_from_hub:
            loaded = self._try_load_from_hub()
            if loaded:
                print("✓ Loaded model from HuggingFace Hub")

        # Build model if not loaded
        if not hasattr(self, 'model'):
            self.model = self._build_model()
            self.optimizer = None
            self.scheduler = None
            self.global_step = 0
            self.epoch = 0
            self.best_loss = float('inf')

        self.loss_fn = self._build_loss_fn()

        # Initialize optimizer if not loaded
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

        # Initialize scheduler if not loaded
        if self.scheduler is None and config.use_scheduler:
            self.scheduler = self._build_scheduler()

        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize prompt generation
        print("Initializing prompt generation...")
        self.prompt_gen = SynthesisSystem(seed=config.seed)
        self.flavors = self._load_flavors()
        self.used_prompts = []
        self.prompt_sources = []

        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                name=f"lyra_sdxl_{config.fusion_strategy}",
                resume="allow"
            )

    def _init_hf_repo(self):
        """Initialize HuggingFace repository."""
        if not self.config.push_to_hub:
            return

        try:
            create_repo(
                self.config.hf_repo,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            print(f"✓ HuggingFace repo: https://huggingface.co/{self.config.hf_repo}")
        except Exception as e:
            print(f"⚠️  Could not create HF repo: {e}")
            print("   (Repo may already exist or you may need to login)")

    def _try_load_from_hub(self) -> bool:
        """Try to load the latest model from HuggingFace Hub."""
        try:
            print(f"🔍 Checking for existing model on HF: {self.config.hf_repo}")

            # Try to download model checkpoint
            try:
                model_path = hf_hub_download(
                    repo_id=self.config.hf_repo,
                    filename="model.pt",
                    repo_type="model"
                )

                checkpoint = torch.load(model_path, map_location=self.device)

                # Build model and load state
                self.model = self._build_model()
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Load optimizer
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load training state
                self.global_step = checkpoint.get('global_step', 0)
                self.epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Load scheduler if exists
                if 'scheduler_state_dict' in checkpoint and self.config.use_scheduler:
                    self.scheduler = self._build_scheduler()
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                print(f"✓ Resumed from step {self.global_step}, epoch {self.epoch}")
                return True

            except Exception as e:
                print(f"   No existing model found: {e}")
                return False

        except Exception as e:
            print(f"⚠️  Could not access HF Hub: {e}")
            return False

    def _push_to_hub(self, is_best: bool = False):
        """Push current model to HuggingFace Hub."""
        if not self.config.push_to_hub:
            return

        try:
            print(f"\n📤 Pushing to HuggingFace Hub...", end=" ", flush=True)

            # Save checkpoint locally first
            temp_path = Path(self.config.checkpoint_dir) / "temp_upload.pt"

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_loss': self.best_loss,
                'config': asdict(self.config)
            }

            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, temp_path)

            # Upload to HF
            self.hf_api.upload_file(
                path_or_fileobj=str(temp_path),
                path_in_repo="model.pt",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Step {self.global_step}: loss={self.best_loss:.4f}" if is_best else f"Training step {self.global_step}"
            )

            # Save config as well
            config_path = Path(self.config.checkpoint_dir) / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)

            self.hf_api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Config update at step {self.global_step}"
            )

            # Create model card if this is the best model
            if is_best:
                self._create_model_card()

            # Cleanup
            temp_path.unlink()

            print(f"✓ Pushed to https://huggingface.co/{self.config.hf_repo}")

        except Exception as e:
            print(f"✗ Failed: {e}")

    def _create_model_card(self):
        """Create/update model card on HuggingFace."""
        model_card = f"""---
tags:
- vae
- multimodal
- text-embeddings
- clip
- t5
- sdxl
- stable-diffusion
license: mit
---

# VAE Lyra 🎵 - SDXL Edition

Multi-modal Variational Autoencoder for SDXL text embedding transformation using geometric fusion.
Fuses CLIP-L, CLIP-G, and T5-XXL into a unified latent space.

## Model Details

- **Fusion Strategy**: {self.config.fusion_strategy}
- **Latent Dimension**: {self.config.latent_dim}
- **Training Steps**: {self.global_step:,}
- **Best Loss**: {self.best_loss:.4f}

## Architecture

- **Modalities**: 
  - CLIP-L (768d) - SDXL text_encoder
  - CLIP-G (1280d) - SDXL text_encoder_2  
  - T5-XXL (2048d) - Additional conditioning
- **Encoder Layers**: {self.config.encoder_layers}
- **Decoder Layers**: {self.config.decoder_layers}
- **Hidden Dimension**: {self.config.hidden_dim}

## SDXL Compatibility

This model outputs both CLIP embeddings needed for SDXL:
- `clip_l`: [batch, 77, 768] → text_encoder output
- `clip_g`: [batch, 77, 1280] → text_encoder_2 output

T5-XXL information is encoded into the latent space but not directly output.

## Usage
```python
from geovocab2.train.model.vae.vae_lyra import MultiModalVAE, MultiModalVAEConfig
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="{self.config.hf_repo}",
    filename="model.pt"
)

# Load checkpoint
checkpoint = torch.load(model_path)

# Create model
config = MultiModalVAEConfig(
    modality_dims={{"clip_l": 768, "clip_g": 1280, "t5_xl": 2048}},
    latent_dim={self.config.latent_dim},
    fusion_strategy="{self.config.fusion_strategy}"
)

model = MultiModalVAE(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use model - train on all three
inputs = {{
    "clip_l": clip_l_embeddings,   # [batch, 77, 768]
    "clip_g": clip_g_embeddings,   # [batch, 77, 1280]
    "t5_xl": t5_xl_embeddings      # [batch, 77, 2048]
}}

# For SDXL inference - only decode CLIP outputs
recons, mu, logvar = model(inputs, target_modalities=["clip_l", "clip_g"])

# Use recons["clip_l"] and recons["clip_g"] with SDXL
```

## Training Details

- Trained on {len(self.used_prompts):,} diverse prompts
- Mix of LAION flavors ({100 * (1 - self.config.synthetic_ratio):.0f}%) and synthetic prompts ({100 * self.config.synthetic_ratio:.0f}%)
- KL Annealing: {self.config.use_kl_annealing}
- Learning Rate: {self.config.learning_rate}

## Citation
```bibtex
@software{{vae_lyra_sdxl_2025,
  author = {{AbstractPhil}},
  title = {{VAE Lyra SDXL: Multi-Modal Variational Autoencoder}},
  year = {{2025}},
  url = {{https://huggingface.co/{self.config.hf_repo}}}
}}
```
"""

        try:
            card_path = Path(self.config.checkpoint_dir) / "README.md"
            with open(card_path, 'w') as f:
                f.write(model_card)

            self.hf_api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=self.config.hf_repo,
                repo_type="model",
                commit_message=f"Update model card (step {self.global_step})"
            )
        except Exception as e:
            print(f"⚠️  Could not update model card: {e}")

    def _build_model(self) -> nn.Module:
        """Build VAE Lyra model."""
        vae_config = MultiModalVAEConfig(
            modality_dims=self.config.modality_dims,
            latent_dim=self.config.latent_dim,
            seq_len=self.config.seq_len,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            fusion_strategy=self.config.fusion_strategy,
            fusion_heads=self.config.fusion_heads,
            fusion_dropout=self.config.fusion_dropout
        )

        model = MultiModalVAE(vae_config)
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ VAE Lyra parameters: {total_params:,}")
        print(f"✓ Fusion strategy: {self.config.fusion_strategy}")

        return model

    def _build_loss_fn(self):
        """Build VAE loss function - now stateless."""
        return MultiModalVAELoss(
            beta_kl=self.config.beta_kl,
            beta_reconstruction=self.config.beta_reconstruction,
            beta_cross_modal=self.config.beta_cross_modal,
            recon_type=self.config.recon_type,
            modality_weights=self.config.modality_recon_weights
            # No modality_dims or projection_dim needed anymore!
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler_type == 'onecycle':
            return None  # Initialized after knowing steps per epoch
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def _load_flavors(self):
        """Load LAION flavors from clip-interrogator."""
        print("Loading LAION flavors...")
        try:
            r = requests.get(
                "https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/flavors.txt",
                timeout=30
            )
            flavors = [line.strip() for line in r.text.split('\n') if line.strip()]
            print(f"✓ Loaded {len(flavors):,} LAION flavors")
            return flavors
        except Exception as e:
            print(f"⚠️  Failed to load flavors: {e}")
            print("Using fallback prompts...")
            return [
                "a beautiful landscape",
                "abstract art",
                "a detailed portrait",
                "futuristic architecture",
                "natural scenery"
            ]

    def _generate_prompt(self):
        """Generate a prompt (synthetic or LAION)."""
        if random.random() < self.config.synthetic_ratio:
            complexity = random.choice([2, 3, 4, 5])
            prompt = self.prompt_gen.synthesize(complexity=complexity)['text']
            source = "synthetic"
        else:
            prompt = random.choice(self.flavors)
            source = "laion"

        return prompt, source

    def _get_current_kl_beta(self) -> float:
        """Get current KL beta with annealing."""
        if not self.config.use_kl_annealing:
            return self.config.beta_kl

        if self.epoch >= self.config.kl_anneal_epochs:
            return self.config.beta_kl

        # Linear annealing
        progress = self.epoch / self.config.kl_anneal_epochs
        current_beta = self.config.kl_start_beta + \
                       (self.config.beta_kl - self.config.kl_start_beta) * progress

        return current_beta

    def prepare_data(self, num_samples: Optional[int] = None) -> DataLoader:
        """
        Prepare dataset with CLIP-L, CLIP-G, and T5-XXL encoders.

        Args:
            num_samples: Number of training samples (default: from config)
        """
        num_samples = num_samples or self.config.num_samples

        print(f"\n🎼 Generating {num_samples:,} training prompts...")

        # Generate prompts
        texts = []
        sources = []
        for _ in tqdm(range(num_samples), desc="Generating prompts"):
            prompt, source = self._generate_prompt()
            texts.append(prompt)
            sources.append(source)

        # Store for logging
        self.used_prompts = texts
        self.prompt_sources = sources

        # Count sources
        source_counts = Counter(sources)
        print(f"\nPrompt distribution:")
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} ({count / num_samples * 100:.1f}%)")

        # Show samples
        print(f"\nSample prompts:")
        for i in range(min(5, len(texts))):
            print(f"  [{sources[i]}] {texts[i]}")

        # Load text encoders
        print("\nLoading CLIP-L, CLIP-G, and T5-XXL...")
        print("  [1/3] CLIP-L (openai/clip-vit-large-patch14)...")
        clip_l_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_l_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        print("  [2/3] CLIP-G (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)...")
        clip_g_tokenizer = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        )
        clip_g_model = CLIPTextModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        )

        print("  [3/3] FLAN-T5-XL (flan-t5-xl)...")
        t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        t5_model = T5EncoderModel.from_pretrained("google/flan-t5-xl")

        print(f"✓ All encoders loaded")
        print(f"Dataset size: {len(texts)} samples")

        # Create dataset
        dataset = TextEmbeddingDataset(
            texts=texts,
            clip_l_tokenizer=clip_l_tokenizer,
            clip_l_model=clip_l_model,
            clip_g_tokenizer=clip_g_tokenizer,
            clip_g_model=clip_g_model,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            device=self.device,
            max_length=self.config.seq_len
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # Initialize OneCycle scheduler if needed
        if self.config.scheduler_type == 'onecycle' and self.scheduler is None:
            steps_per_epoch = len(dataloader)
            total_steps = steps_per_epoch * self.config.num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.3
            )

        return dataloader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with three modalities."""
        modality_inputs = {
            'clip_l': batch['clip_l'].to(self.device),
            'clip_g': batch['clip_g'].to(self.device),
            't5_xl': batch['t5_xl'].to(self.device)
        }

        # Update KL beta with annealing
        current_kl_beta = self._get_current_kl_beta()
        self.loss_fn.beta_kl = current_kl_beta

        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                # Train on all three modalities
                reconstructions, mu, logvar = self.model(modality_inputs)

                # Project to common space for cross-modal loss
                projected_recons = self.model.project_for_cross_modal(reconstructions)

                loss, components = self.loss_fn(
                    inputs=modality_inputs,
                    reconstructions=reconstructions,
                    mu=mu,
                    logvar=logvar,
                    projected_recons=projected_recons,  # <-- Pass projections
                    return_components=True
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            reconstructions, mu, logvar = self.model(modality_inputs)
            projected_recons = self.model.project_for_cross_modal(reconstructions)

            loss, components = self.loss_fn(
                inputs=modality_inputs,
                reconstructions=reconstructions,
                mu=mu,
                logvar=logvar,
                projected_recons=projected_recons,
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

        if self.scheduler is not None and self.config.scheduler_type == 'onecycle':
            self.scheduler.step()

        metrics = {k: v.item() for k, v in components.items()}
        metrics['kl_beta'] = current_kl_beta

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}

        pbar = tqdm(dataloader, desc=f"🎵 Epoch {self.epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            self.global_step += 1

            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'r_l': f"{metrics.get('recon_clip_l', 0):.4f}",
                'r_g': f"{metrics.get('recon_clip_g', 0):.4f}",
                'kl': f"{metrics['kl']:.4f}",
                'β': f"{metrics['kl_beta']:.3f}"
            })

            if self.config.use_wandb and self.global_step % self.config.log_every == 0:
                wandb.log({
                    **{f'train/{k}': v for k, v in metrics.items()},
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.epoch,
                    'train/step': self.global_step
                })

            # Push to Hub at intervals
            if self.global_step % self.config.push_every == 0:
                self._push_to_hub()

            # Save checkpoint at intervals
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        return avg_metrics

    def train(self, dataloader: DataLoader):
        """Full training loop."""
        print(f"\n{'=' * 70}")
        print(f"🎵 Starting VAE Lyra SDXL training for {self.config.num_epochs} epochs...")
        print(f"{'=' * 70}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total steps per epoch: {len(dataloader)}")
        print(f"Total training samples: {len(self.used_prompts):,}")
        print(f"Fusion strategy: {self.config.fusion_strategy}")
        print(f"KL annealing: {self.config.use_kl_annealing}")
        print(f"Push to HF every: {self.config.push_every} steps")

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            metrics = self.train_epoch(dataloader)

            # Update scheduler (for Cosine)
            if self.scheduler is not None and self.config.scheduler_type == 'cosine':
                self.scheduler.step()

            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Total Loss: {metrics['total']:.4f}")
            print(f"  Reconstruction (CLIP-L): {metrics.get('recon_clip_l', 0):.4f}")
            print(f"  Reconstruction (CLIP-G): {metrics.get('recon_clip_g', 0):.4f}")
            print(f"  Reconstruction (T5-XXL): {metrics.get('recon_t5_xl', 0):.4f}")
            print(f"  KL Divergence: {metrics['kl']:.4f}")
            print(f"  Cross-Modal: {metrics.get('cross_modal', 0):.4f}")
            print(f"  KL Beta: {metrics['kl_beta']:.3f}")
            print(f"{'=' * 70}")

            if metrics['total'] < self.best_loss:
                self.best_loss = metrics['total']
                self.save_checkpoint(best=True)
                self._push_to_hub(is_best=True)
                print(f"  ✨ New best model saved and pushed! Loss: {self.best_loss:.4f}")

        print("\n✨ Training complete!")
        print(f"📤 Final model at: https://huggingface.co/{self.config.hf_repo}")

        if self.config.use_wandb:
            wandb.finish()

    def save_checkpoint(self, best: bool = False):
        """Save model checkpoint with prompt history."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss,
            'used_prompts': self.used_prompts,
            'prompt_sources': self.prompt_sources
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if best:
            path = Path(self.config.checkpoint_dir) / 'best_model.pt'
        else:
            path = Path(self.config.checkpoint_dir) / f'checkpoint_step_{self.global_step}.pt'

        torch.save(checkpoint, path)

        # Save prompts as text file
        if best:
            prompts_path = Path(self.config.checkpoint_dir) / 'training_prompts.txt'
            with open(prompts_path, 'w') as f:
                for prompt, source in zip(self.used_prompts, self.prompt_sources):
                    f.write(f"[{source}] {prompt}\n")

        if not best:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob('checkpoint_step_*.pt')],
            key=lambda x: int(x.stem.split('_')[-1])
        )

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

        if 'used_prompts' in checkpoint:
            self.used_prompts = checkpoint['used_prompts']
            self.prompt_sources = checkpoint.get('prompt_sources', [])

        print(f"✓ Loaded checkpoint from step {self.global_step}")
        print(f"✓ Best loss: {self.best_loss:.4f}")
        if self.used_prompts:
            print(f"✓ Training prompts: {len(self.used_prompts):,}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_lyra_trainer(
        fusion_strategy: str = "cantor",
        num_samples: int = 10000,
        batch_size: int = 16,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        beta_kl: float = 0.1,
        use_kl_annealing: bool = True,
        push_to_hub: bool = True,
        push_every: int = 2000,
        hf_repo: str = "AbstractPhil/vae-lyra-sdxl",
        **kwargs
) -> VAELyraTrainer:
    """
    Convenience function to create VAE Lyra SDXL trainer.

    Example:
    #    >>> trainer = create_lyra_trainer(
    #    ...     fusion_strategy="cantor",
    #    ...     num_samples=10000,
    #    ...     batch_size=16,
    #    ...     push_to_hub=True
    #    ... )
    """
    config = VAELyraTrainerConfig(
        fusion_strategy=fusion_strategy,
        num_samples=num_samples,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        beta_kl=beta_kl,
        use_kl_annealing=use_kl_annealing,
        push_to_hub=push_to_hub,
        push_every=push_every,
        hf_repo=hf_repo,
        **kwargs
    )
    return VAELyraTrainer(config)


def load_lyra_from_hub(
        repo_id: str = "AbstractPhil/vae-lyra-sdxl",
        device: str = "cuda"
) -> MultiModalVAE:
    """
    Load VAE Lyra SDXL directly from HuggingFace Hub.

    Example:
    #    >>> model = load_lyra_from_hub()
    #    >>> model.eval()
    #"""
    from huggingface_hub import hf_hub_download

    # Download model
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        repo_type="model"
    )

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        repo_type="model"
    )

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    # Create VAE config
    vae_config = MultiModalVAEConfig(
        modality_dims=config_dict.get('modality_dims', {"clip_l": 768, "clip_g": 1280, "t5_xl": 2048}),
        latent_dim=config_dict.get('latent_dim', 768),
        seq_len=config_dict.get('seq_len', 77),
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),
        fusion_strategy=config_dict.get('fusion_strategy', 'cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1)
    )

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiModalVAE(vae_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✓ Loaded VAE Lyra SDXL from {repo_id}")
    print(f"✓ Training step: {checkpoint.get('global_step', 'unknown')}")
    print(f"✓ Best loss: {checkpoint.get('best_loss', 'unknown'):.4f}")

    return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create trainer for SDXL
    trainer = create_lyra_trainer(
        fusion_strategy="cantor",  # or "geometric"
        num_samples=10000,
        batch_size=16,
        num_epochs=100,
        push_to_hub=True,
        hf_repo="AbstractPhil/vae-lyra-sdxl"
    )

    # Prepare data
    dataloader = trainer.prepare_data()

    # Train
    trainer.train(dataloader)