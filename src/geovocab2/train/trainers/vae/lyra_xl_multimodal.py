# geovocab2/train/trainers/vae_lyra/trainer.py

"""
Trainer for VAE Lyra - Multi-Modal Variational Autoencoder
CLIP-L + CLIP-G + T5-XL (Decoupled) for SDXL Compatibility

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
    FusionStrategy
)
from geovocab2.data.prompt.symbolic_tree import SynthesisSystem


@dataclass
class VAELyraTrainerConfig:
    """Training configuration for VAE Lyra with SDXL support and decoupled T5."""

    # Model architecture - SDXL with decoupled T5
    modality_dims: Dict[str, int] = None  # {"clip_l": 768, "clip_g": 1280, "t5_xl_l": 2048, "t5_xl_g": 2048}
    modality_seq_lens: Dict[str, int] = None  # {"clip_l": 77, "clip_g": 77, "t5_xl_l": 512, "t5_xl_g": 512}
    binding_config: Dict[str, Dict[str, float]] = None  # T5 binding configuration

    latent_dim: int = 2048
    seq_len: int = 77
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.1

    # Fusion
    fusion_strategy: str = "adaptive_cantor"  # Default to adaptive
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    cantor_depth: int = 8
    cantor_local_window: int = 3

    # Adaptive fusion parameters
    alpha_init: float = 1.0
    beta_init: float = 0.3
    alpha_lr_scale: float = 0.1
    beta_lr_scale: float = 1.0

    # Loss weights
    beta_kl: float = 0.1
    beta_reconstruction: float = 1.0
    beta_cross_modal: float = 0.05
    beta_alpha_regularization: float = 0.01
    recon_type: str = 'mse'

    # Per-modality reconstruction weights
    modality_recon_weights: Dict[str, float] = None

    # KL annealing
    use_kl_annealing: bool = True
    kl_anneal_epochs: int = 10
    kl_start_beta: float = 0.0

    # Training hyperparameters
    batch_size: int = 8  # Reduced for larger models with longer sequences
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
    checkpoint_dir: str = './checkpoints_lyra_adaptive'
    save_every: int = 1000
    keep_last_n: int = 3

    # HuggingFace Hub
    hf_repo: str = "AbstractPhil/vae-lyra-adaptive-cantor"
    push_to_hub: bool = True
    push_every: int = 2000
    auto_load_from_hub: bool = True

    # Logging
    use_wandb: bool = True
    wandb_project: str = 'vae-lyra-adaptive-cantor'
    wandb_entity: Optional[str] = None
    log_every: int = 50

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True

    # Misc
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        # Default: SDXL configuration with decoupled T5
        if self.modality_dims is None:
            self.modality_dims = {
                "clip_l": 768,
                "clip_g": 1280,
                "t5_xl_l": 2048,  # T5 for CLIP-L
                "t5_xl_g": 2048  # T5 for CLIP-G
            }

        # Default: Different sequence lengths
        if self.modality_seq_lens is None:
            self.modality_seq_lens = {
                "clip_l": 77,
                "clip_g": 77,
                "t5_xl_l": 512,  # Longer context for T5
                "t5_xl_g": 512
            }

        # Default binding: Decoupled T5 scales
        if self.binding_config is None:
            self.binding_config = {
                "clip_l": {"t5_xl_l": 0.3},  # CLIP-L uses T5-L scale
                "clip_g": {"t5_xl_g": 0.3},  # CLIP-G uses T5-G scale
                "t5_xl_l": {},  # T5-L stays independent
                "t5_xl_g": {}  # T5-G stays independent
            }

        # Default weights: prioritize CLIP outputs for SDXL
        if self.modality_recon_weights is None:
            self.modality_recon_weights = {
                "clip_l": 1.0,  # Full weight - SDXL primary
                "clip_g": 1.0,  # Full weight - SDXL primary
                "t5_xl_l": 0.3,  # Lower weight - auxiliary signal
                "t5_xl_g": 0.3  # Lower weight - auxiliary signal
            }


class TextEmbeddingDataset(Dataset):
    """Dataset that generates CLIP-L, CLIP-G, and decoupled T5-XL embeddings on-the-fly."""

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
            clip_max_length: int = 77,
            t5_max_length: int = 512
    ):
        self.texts = texts
        self.clip_l_tokenizer = clip_l_tokenizer
        self.clip_l_model = clip_l_model
        self.clip_g_tokenizer = clip_g_tokenizer
        self.clip_g_model = clip_g_model
        self.t5_tokenizer = t5_tokenizer
        self.t5_model = t5_model
        self.device = device
        self.clip_max_length = clip_max_length
        self.t5_max_length = t5_max_length

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
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_l_output = self.clip_l_model(**clip_l_tokens)
        clip_l_embed = clip_l_output.last_hidden_state.squeeze(0)

        # CLIP-G embedding (SDXL text_encoder_2)
        clip_g_tokens = self.clip_g_tokenizer(
            text,
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        clip_g_output = self.clip_g_model(**clip_g_tokens)
        clip_g_embed = clip_g_output.last_hidden_state.squeeze(0)

        # T5-XL embeddings (longer sequence for richer context)
        t5_tokens = self.t5_tokenizer(
            text,
            max_length=self.t5_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        t5_output = self.t5_model(**t5_tokens)
        t5_embed = t5_output.last_hidden_state.squeeze(0)

        # Duplicate T5 for decoupled scales (same embedding, different learned binding)
        return {
            'clip_l': clip_l_embed.cpu(),
            'clip_g': clip_g_embed.cpu(),
            't5_xl_l': t5_embed.cpu(),  # T5 scale for CLIP-L
            't5_xl_g': t5_embed.cpu(),  # T5 scale for CLIP-G (same input, different learned path)
            'text': text
        }


class VAELyraTrainer:
    """Trainer for VAE Lyra - SDXL-compatible Multi-Modal VAE with Adaptive Cantor Fusion."""

    def __init__(self, config: VAELyraTrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # HuggingFace API
        self.hf_api = HfApi()
        self._init_hf_repo()

        print("🎵 Initializing VAE Lyra (Adaptive Cantor Edition)...")
        print(f"   Modalities: CLIP-L (768@77), CLIP-G (1280@77)")
        print(f"   T5-XL-L (2048@512), T5-XL-G (2048@512)")
        print(f"   Fusion: {config.fusion_strategy}")

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
            self.optimizer = self._build_optimizer()

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
                name=f"lyra_{config.fusion_strategy}_alpha{config.alpha_init}_beta{config.beta_init}",
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
                self.optimizer = self._build_optimizer()
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

        # Get fusion parameters if available
        fusion_params_str = ""
        if hasattr(self.model, 'get_fusion_params'):
            params = self.model.get_fusion_params()
            if params:
                fusion_params_str = "\n## Learned Parameters\n\n"
                if 'alphas' in params:
                    fusion_params_str += "**Alpha (Visibility):**\n"
                    for name, alpha in params['alphas'].items():
                        fusion_params_str += f"- {name}: {torch.sigmoid(alpha).item():.4f}\n"
                if 'betas' in params:
                    fusion_params_str += "\n**Beta (Capacity):**\n"
                    for name, beta in params['betas'].items():
                        fusion_params_str += f"- {name}: {torch.sigmoid(beta).item():.4f}\n"

        model_card = f"""---
tags:
- vae
- multimodal
- text-embeddings
- clip
- t5
- sdxl
- stable-diffusion
- adaptive-cantor
- geometric-fusion
license: mit
---

# VAE Lyra 🎵 - Adaptive Cantor Edition

Multi-modal Variational Autoencoder for SDXL text embedding transformation using adaptive Cantor fractal fusion with learned alpha (visibility) and beta (capacity) parameters.

Fuses CLIP-L, CLIP-G, and decoupled T5-XL scales into a unified latent space.

## Model Details

- **Fusion Strategy**: {self.config.fusion_strategy}
- **Latent Dimension**: {self.config.latent_dim}
- **Training Steps**: {self.global_step:,}
- **Best Loss**: {self.best_loss:.4f}
{fusion_params_str}

## Architecture

- **Modalities** (with sequence lengths): 
  - CLIP-L (768d @ 77 tokens) - SDXL text_encoder
  - CLIP-G (1280d @ 77 tokens) - SDXL text_encoder_2  
  - T5-XL-L (2048d @ 512 tokens) - Auxiliary for CLIP-L
  - T5-XL-G (2048d @ 512 tokens) - Auxiliary for CLIP-G
- **Encoder Layers**: {self.config.encoder_layers}
- **Decoder Layers**: {self.config.decoder_layers}
- **Hidden Dimension**: {self.config.hidden_dim}
- **Cantor Depth**: {self.config.cantor_depth}
- **Local Window**: {self.config.cantor_local_window}

## Key Features

### Adaptive Cantor Fusion
- **Cantor Fractal Routing**: Sparse attention based on fractal coordinate mapping
- **Learned Alpha (Visibility)**: Per-modality parameters controlling latent space usage (tied to KL divergence)
- **Learned Beta (Capacity)**: Per-binding-pair parameters controlling source influence strength

### Decoupled T5 Scales
- T5-XL-L binds specifically to CLIP-L (weight: {self.config.binding_config.get('clip_l', {}).get('t5_xl_l', 0.3)})
- T5-XL-G binds specifically to CLIP-G (weight: {self.config.binding_config.get('clip_g', {}).get('t5_xl_g', 0.3)})
- Independent T5 representations allow specialized semantic enrichment per CLIP encoder

### Variable Sequence Lengths
- CLIP: 77 tokens (standard)
- T5: 512 tokens (extended context for richer semantic capture)

## SDXL Compatibility

This model outputs both CLIP embeddings needed for SDXL:
- `clip_l`: [batch, 77, 768] → text_encoder output
- `clip_g`: [batch, 77, 1280] → text_encoder_2 output

T5 information is encoded into the latent space and influences both CLIP outputs through learned binding weights.

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
    modality_dims={{
        "clip_l": 768,
        "clip_g": 1280,
        "t5_xl_l": 2048,
        "t5_xl_g": 2048
    }},
    modality_seq_lens={{
        "clip_l": 77,
        "clip_g": 77,
        "t5_xl_l": 512,
        "t5_xl_g": 512
    }},
    binding_config={{
        "clip_l": {{"t5_xl_l": 0.3}},
        "clip_g": {{"t5_xl_g": 0.3}},
        "t5_xl_l": {{}},
        "t5_xl_g": {{}}
    }},
    latent_dim={self.config.latent_dim},
    fusion_strategy="{self.config.fusion_strategy}",
    cantor_depth={self.config.cantor_depth},
    cantor_local_window={self.config.cantor_local_window}
)

model = MultiModalVAE(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use model - train on all four modalities
inputs = {{
    "clip_l": clip_l_embeddings,     # [batch, 77, 768]
    "clip_g": clip_g_embeddings,     # [batch, 77, 1280]
    "t5_xl_l": t5_xl_l_embeddings,   # [batch, 512, 2048]
    "t5_xl_g": t5_xl_g_embeddings    # [batch, 512, 2048]
}}

# For SDXL inference - only decode CLIP outputs
recons, mu, logvar, per_mod_mus = model(inputs, target_modalities=["clip_l", "clip_g"])

# Use recons["clip_l"] and recons["clip_g"] with SDXL
```

## Training Details

- Trained on {len(self.used_prompts):,} diverse prompts
- Mix of LAION flavors ({100 * (1 - self.config.synthetic_ratio):.0f}%) and synthetic prompts ({100 * self.config.synthetic_ratio:.0f}%)
- KL Annealing: {self.config.use_kl_annealing}
- Learning Rate: {self.config.learning_rate}
- Alpha Init: {self.config.alpha_init}
- Beta Init: {self.config.beta_init}

## Citation
```bibtex
@software{{vae_lyra_adaptive_cantor_2025,
  author = {{AbstractPhil}},
  title = {{VAE Lyra: Adaptive Cantor Multi-Modal Variational Autoencoder}},
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
        """Build VAE Lyra model with adaptive fusion."""
        vae_config = MultiModalVAEConfig(
            modality_dims=self.config.modality_dims,
            modality_seq_lens=self.config.modality_seq_lens,
            binding_config=self.config.binding_config,
            latent_dim=self.config.latent_dim,
            seq_len=self.config.seq_len,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            fusion_strategy=self.config.fusion_strategy,
            fusion_heads=self.config.fusion_heads,
            fusion_dropout=self.config.fusion_dropout,
            cantor_depth=self.config.cantor_depth,
            cantor_local_window=self.config.cantor_local_window,
            alpha_init=self.config.alpha_init,
            beta_init=self.config.beta_init,
            alpha_lr_scale=self.config.alpha_lr_scale,
            beta_lr_scale=self.config.beta_lr_scale,
            beta_alpha_regularization=self.config.beta_alpha_regularization,
            seed=self.config.seed
        )

        model = MultiModalVAE(vae_config)
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ VAE Lyra parameters: {total_params:,}")
        print(f"✓ Fusion strategy: {self.config.fusion_strategy}")

        # Print learned parameters info
        if hasattr(model, 'get_fusion_params'):
            params = model.get_fusion_params()
            if params:
                print(f"✓ Adaptive parameters:")
                print(f"   Alpha (visibility): {len(params.get('alphas', {}))}")
                print(f"   Beta (capacity): {len(params.get('betas', {}))}")

        return model

    def _build_optimizer(self):
        """Build optimizer with different learning rates for alpha/beta."""
        # Separate parameters for different learning rates
        param_groups = []

        # Regular parameters
        regular_params = []
        alpha_params = []
        beta_params = []

        for name, param in self.model.named_parameters():
            if 'alphas' in name:
                alpha_params.append(param)
            elif 'betas' in name:
                beta_params.append(param)
            else:
                regular_params.append(param)

        # Regular parameters - standard learning rate
        if regular_params:
            param_groups.append({
                'params': regular_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })

        # Alpha parameters - scaled learning rate
        if alpha_params:
            param_groups.append({
                'params': alpha_params,
                'lr': self.config.learning_rate * self.config.alpha_lr_scale,
                'weight_decay': 0.0  # No weight decay for learned scalars
            })

        # Beta parameters - scaled learning rate
        if beta_params:
            param_groups.append({
                'params': beta_params,
                'lr': self.config.learning_rate * self.config.beta_lr_scale,
                'weight_decay': 0.0  # No weight decay for learned scalars
            })

        return AdamW(param_groups)

    def _build_loss_fn(self):
        """Build VAE loss function."""
        return MultiModalVAELoss(
            beta_kl=self.config.beta_kl,
            beta_reconstruction=self.config.beta_reconstruction,
            beta_cross_modal=self.config.beta_cross_modal,
            beta_alpha_regularization=self.config.beta_alpha_regularization,
            recon_type=self.config.recon_type,
            modality_weights=self.config.modality_recon_weights
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
        Prepare dataset with CLIP-L, CLIP-G, and decoupled T5-XL encoders.

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
        print("\nLoading CLIP-L, CLIP-G, and T5-XL...")
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

        print("  [3/3] FLAN-T5-XL (google/flan-t5-xl)...")
        t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        t5_model = T5EncoderModel.from_pretrained("google/flan-t5-xl")

        print(f"✓ All encoders loaded")
        print(f"Dataset size: {len(texts)} samples")

        # Create dataset with longer T5 sequences
        dataset = TextEmbeddingDataset(
            texts=texts,
            clip_l_tokenizer=clip_l_tokenizer,
            clip_l_model=clip_l_model,
            clip_g_tokenizer=clip_g_tokenizer,
            clip_g_model=clip_g_model,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            device=self.device,
            clip_max_length=self.config.modality_seq_lens['clip_l'],
            t5_max_length=self.config.modality_seq_lens['t5_xl_l']
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
        """Single training step with four modalities."""
        modality_inputs = {
            'clip_l': batch['clip_l'].to(self.device),
            'clip_g': batch['clip_g'].to(self.device),
            't5_xl_l': batch['t5_xl_l'].to(self.device),
            't5_xl_g': batch['t5_xl_g'].to(self.device)
        }

        # Update KL beta with annealing
        current_kl_beta = self._get_current_kl_beta()
        self.loss_fn.beta_kl = current_kl_beta

        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                # Forward pass
                reconstructions, mu, logvar, per_mod_mus = self.model(modality_inputs)

                # Get fusion parameters for alpha regularization
                fusion_params = self.model.get_fusion_params()
                alphas = fusion_params.get('alphas', None)

                # Project to common space for cross-modal loss
                projected_recons = self.model.project_for_cross_modal(reconstructions)

                loss, components = self.loss_fn(
                    inputs=modality_inputs,
                    reconstructions=reconstructions,
                    mu=mu,
                    logvar=logvar,
                    per_modality_mus=per_mod_mus,
                    alphas=alphas,
                    projected_recons=projected_recons,
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
            reconstructions, mu, logvar, per_mod_mus = self.model(modality_inputs)
            fusion_params = self.model.get_fusion_params()
            alphas = fusion_params.get('alphas', None)
            projected_recons = self.model.project_for_cross_modal(reconstructions)

            loss, components = self.loss_fn(
                inputs=modality_inputs,
                reconstructions=reconstructions,
                mu=mu,
                logvar=logvar,
                per_modality_mus=per_mod_mus,
                alphas=alphas,
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

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in components.items()}
        metrics['kl_beta'] = current_kl_beta

        # Add beta parameters to metrics
        betas = fusion_params.get('betas', {})
        for name, beta in betas.items():
            metrics[f'beta_{name}'] = torch.sigmoid(beta).item()

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
                'α_l': f"{metrics.get('alpha_clip_l', 0):.3f}",
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
        print(f"🎵 Starting VAE Lyra Adaptive Cantor training for {self.config.num_epochs} epochs...")
        print(f"{'=' * 70}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total steps per epoch: {len(dataloader)}")
        print(f"Total training samples: {len(self.used_prompts):,}")
        print(f"Fusion strategy: {self.config.fusion_strategy}")
        print(f"KL annealing: {self.config.use_kl_annealing}")
        print(f"Push to HF every: {self.config.push_every} steps")
        print(f"Alpha init: {self.config.alpha_init}, Beta init: {self.config.beta_init}")

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
            print(f"  Reconstruction (T5-XL-L): {metrics.get('recon_t5_xl_l', 0):.4f}")
            print(f"  Reconstruction (T5-XL-G): {metrics.get('recon_t5_xl_g', 0):.4f}")
            print(f"  KL Divergence: {metrics['kl']:.4f}")
            print(f"  Cross-Modal: {metrics.get('cross_modal', 0):.4f}")
            print(f"  Alpha Regularization: {metrics.get('alpha_reg', 0):.4f}")

            # Print learned alpha values
            alpha_str = ", ".join([f"{k.split('_')[1]}: {v:.3f}"
                                   for k, v in metrics.items() if k.startswith('alpha_') and k != 'alpha_reg'])
            if alpha_str:
                print(f"  Learned Alphas: {alpha_str}")

            # Print learned beta values
            beta_str = ", ".join([f"{k.replace('beta_', '')}: {v:.3f}"
                                  for k, v in metrics.items() if k.startswith('beta_') and '_' in k[5:]])
            if beta_str:
                print(f"  Learned Betas: {beta_str}")

            print(f"  KL Beta (schedule): {metrics['kl_beta']:.3f}")
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
        fusion_strategy: str = "adaptive_cantor",
        num_samples: int = 10000,
        batch_size: int = 8,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        beta_kl: float = 0.1,
        alpha_init: float = 1.0,
        beta_init: float = 0.3,
        use_kl_annealing: bool = True,
        push_to_hub: bool = True,
        push_every: int = 2000,
        hf_repo: str = "AbstractPhil/vae-lyra-adaptive-cantor",
        **kwargs
) -> VAELyraTrainer:
    """
    Convenience function to create VAE Lyra Adaptive Cantor trainer.

    Example:
        >>> trainer = create_lyra_trainer(
        ...     fusion_strategy="adaptive_cantor",
        ...     num_samples=10000,
        ...     batch_size=8,
        ...     alpha_init=1.0,
        ...     beta_init=0.3,
        ...     push_to_hub=True
        ... )
    """
    config = VAELyraTrainerConfig(
        fusion_strategy=fusion_strategy,
        num_samples=num_samples,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        beta_kl=beta_kl,
        alpha_init=alpha_init,
        beta_init=beta_init,
        use_kl_annealing=use_kl_annealing,
        push_to_hub=push_to_hub,
        push_every=push_every,
        hf_repo=hf_repo,
        **kwargs
    )
    return VAELyraTrainer(config)


def load_lyra_from_hub(
        repo_id: str = "AbstractPhil/vae-lyra-adaptive-cantor",
        device: str = "cuda"
) -> MultiModalVAE:
    """
    Load VAE Lyra Adaptive Cantor directly from HuggingFace Hub.

    Example:
        >>> model = load_lyra_from_hub()
        >>> model.eval()
    """
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
        modality_dims=config_dict.get('modality_dims'),
        modality_seq_lens=config_dict.get('modality_seq_lens'),
        binding_config=config_dict.get('binding_config'),
        latent_dim=config_dict.get('latent_dim', 2048),
        seq_len=config_dict.get('seq_len', 77),
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),
        fusion_strategy=config_dict.get('fusion_strategy', 'adaptive_cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1),
        cantor_depth=config_dict.get('cantor_depth', 8),
        cantor_local_window=config_dict.get('cantor_local_window', 3),
        alpha_init=config_dict.get('alpha_init', 1.0),
        beta_init=config_dict.get('beta_init', 0.3)
    )

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiModalVAE(vae_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✓ Loaded VAE Lyra Adaptive Cantor from {repo_id}")
    print(f"✓ Training step: {checkpoint.get('global_step', 'unknown')}")
    print(f"✓ Best loss: {checkpoint.get('best_loss', 'unknown'):.4f}")

    return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create trainer with adaptive Cantor fusion
    trainer = create_lyra_trainer(
        fusion_strategy="adaptive_cantor",
        num_samples=10000,
        batch_size=8,
        num_epochs=100,
        alpha_init=1.0,
        beta_init=0.3,
        push_to_hub=True,
        hf_repo="AbstractPhil/vae-lyra-adaptive-cantor"
    )

    # Prepare data
    dataloader = trainer.prepare_data()

    # Train
    trainer.train(dataloader)