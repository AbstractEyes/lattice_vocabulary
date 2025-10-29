"""
Example: SD1.5 Flow Matching with David-Driven Block Penalties

This demonstrates how to use the modular training system to replicate
the functionality of the monolithic trainer.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Dict

# Core training components
from geovocab2.train.trainers.trainer_core import LossComposer, TrainerConfig
from geovocab2.train.trainers.sd15.core.arbitrator import Arbitrator, ArbitratorConfig
from geovocab2.train.trainers.sd15.core.checkpointing import Checkpointer, CheckpointConfig
from geovocab2.train.trainers.sd15.core.assistant_model import DavidAssistant
from geovocab2.train.trainers.sd15.core.teacher_model import SD15Teacher
from geovocab2.train.trainers.sd15.core.student_model import SD15Student, feature_distillation_loss

# External deps (from your repo and HF)
from diffusers import StableDiffusionPipeline, DDPMScheduler
from geovocab2.train.model.core.geo_david_collective import GeoDavidCollective
from geovocab2.data.prompt.symbolic_tree import SynthesisSystem
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import json
import random
from pathlib import Path


# =====================================================================================
# Configuration
# =====================================================================================

@dataclass
class SD15FlowConfig:
    """Complete configuration for SD15 flow matching training."""

    # Run metadata
    run_name: str = "sd15_flowmatch_david"
    out_dir: str = "./runs"
    ckpt_dir: str = "./checkpoints"

    # Models
    model_id: str = "runwayml/stable-diffusion-v1-5"
    david_repo_id: str = "AbstractPhil/geo-david-collective-sd15-base-e40"
    david_cache_dir: str = "./_hf_david_cache"

    # Architecture
    active_blocks: tuple = ("down_0","down_1","down_2","down_3","mid","up_0","up_1","up_2","up_3")
    pooling: str = "mean"
    use_local_flow_heads: bool = False

    # Data
    num_samples: int = 200_000
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42

    # Training
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    amp: bool = True

    # Loss weights
    global_flow_weight: float = 1.0
    block_penalty_weight: float = 0.125
    kd_weight: float = 0.25
    local_flow_weight: float = 1.0

    # David fusion parameters
    alpha_timestep: float = 0.5
    beta_pattern: float = 0.25
    delta_incoherence: float = 0.25
    lambda_min: float = 0.5
    lambda_max: float = 3.0

    # Block base weights (overridden by David config if present)
    block_weights: Dict[str, float] = None

    # Scheduler
    num_train_timesteps: int = 1000

    # Checkpointing
    save_every: int = 1
    save_to_hf: bool = False

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        if self.block_weights is None:
            self.block_weights = {
                'down_0': 0.7, 'down_1': 0.9, 'down_2': 1.0, 'down_3': 1.1,
                'mid': 1.2,
                'up_0': 1.1, 'up_1': 1.0, 'up_2': 0.9, 'up_3': 0.7
            }


# =====================================================================================
# Data
# =====================================================================================

class SymbolicPromptDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, seed: int = 42):
        self.n = n
        random.seed(seed)
        self.sys = SynthesisSystem(seed=seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        r = self.sys.synthesize(complexity=random.choice([1, 2, 3, 4, 5]))
        prompt = r['text']
        t = random.randint(0, 999)
        return {"prompt": prompt, "t": t}


def collate_fn(batch):
    prompts = [b["prompt"] for b in batch]
    t = torch.tensor([b["t"] for b in batch], dtype=torch.long)
    return {"prompts": prompts, "t": t}


def create_data_factory(cfg: SD15FlowConfig):
    """Factory function that creates a DataLoader."""
    def factory():
        dataset = SymbolicPromptDataset(cfg.num_samples, cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    return factory


# =====================================================================================
# David Loading
# =====================================================================================

def load_david(cfg: SD15FlowConfig, device: str) -> GeoDavidCollective:
    """Load David from HuggingFace and configure."""
    # Download repo
    repo_dir = snapshot_download(
        repo_id=cfg.david_repo_id,
        local_dir=cfg.david_cache_dir,
        local_dir_use_symlinks=False
    )

    # Load config
    config_path = Path(repo_dir) / "config.json"
    with open(config_path, "r") as f:
        hf_config = json.load(f)

    # Create David
    david = GeoDavidCollective(
        block_configs=hf_config["block_configs"],
        num_timestep_bins=int(hf_config["num_timestep_bins"]),
        num_patterns_per_bin=int(hf_config["num_patterns_per_bin"]),
        block_weights=hf_config.get("block_weights", {}),
        loss_config=hf_config.get("loss_config", {})
    ).to(device)

    # Load weights
    weights_path = Path(repo_dir) / "model.safetensors"
    state = load_file(str(weights_path))
    david.load_state_dict(state, strict=False)

    print(f"✓ David loaded from {cfg.david_repo_id}")

    # Override block weights if provided in David config
    if "block_weights" in hf_config:
        cfg.block_weights = hf_config["block_weights"]

    return david


# =====================================================================================
# Loss Functions
# =====================================================================================

def create_flow_loss(cfg: SD15FlowConfig):
    """Create global flow matching loss."""
    def loss_fn(batch, outputs, teacher, student, **kwargs):
        v_pred = outputs['student'][0]  # (v_pred, features)
        v_target = outputs['teacher_v_target']

        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        return loss, {'flow_mse': float(loss.item())}

    return loss_fn


def create_kd_loss(cfg: SD15FlowConfig):
    """Create knowledge distillation loss."""
    def loss_fn(batch, outputs, **kwargs):
        s_features = outputs['student'][1]
        t_features = outputs['teacher'][1]

        loss, breakdown = feature_distillation_loss(
            s_features, t_features,
            block_weights=cfg.block_weights,
            pooling=cfg.pooling,
            loss_type='cosine'
        )

        return loss, breakdown

    return loss_fn


def create_local_flow_loss(cfg: SD15FlowConfig):
    """Create local flow head loss."""
    def loss_fn(batch, outputs, student, **kwargs):
        if not cfg.use_local_flow_heads:
            return torch.zeros((), device=batch['t'].device), {}

        s_features = outputs['student'][1]
        v_target = outputs['teacher_v_target']

        loss, breakdown = student.compute_local_flow_loss(
            s_features, v_target, cfg.local_flow_weight
        )

        return loss, breakdown

    return loss_fn


def create_david_penalty_loss(cfg: SD15FlowConfig, assistant: DavidAssistant):
    """Create David-driven adaptive block penalty loss."""
    def loss_fn(batch, outputs, **kwargs):
        s_features = outputs['student'][1]
        t_features = outputs['teacher'][1]
        timesteps = batch['t']

        # Get adaptive weights from David
        adaptive_weights = assistant.compute_adaptive_weights(
            features=s_features,
            timesteps=timesteps,
            base_weights=cfg.block_weights,
            alpha=cfg.alpha_timestep,
            beta=cfg.beta_pattern,
            delta=cfg.delta_incoherence,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max
        )

        # Compute weighted KD loss per block
        loss, breakdown = feature_distillation_loss(
            s_features, t_features,
            block_weights=adaptive_weights,
            pooling=cfg.pooling,
            loss_type='cosine'
        )

        # Add lambda info to breakdown
        for name, lam in adaptive_weights.items():
            breakdown[f'lambda_{name}'] = lam

        return loss, breakdown

    return loss_fn


# =====================================================================================
# Custom Forward Function
# =====================================================================================

def create_forward_fn(teacher: SD15Teacher, student: SD15Student, cfg: SD15FlowConfig):
    """Create custom forward function for training."""
    def forward(batch, **kwargs):
        prompts = batch['prompts']
        t = batch['t']

        # Encode text (shared)
        with torch.no_grad():
            encoder_hidden_states = teacher.encode_text(prompts)

        # Create noisy latents
        device = t.device
        x_t = torch.randn(len(prompts), 4, 64, 64, device=device, dtype=torch.float16)

        # Teacher forward
        with torch.no_grad():
            eps_pred, t_features = teacher.forward_with_features(
                x_t.half(), t, encoder_hidden_states
            )
            # Compute v-target
            v_target = teacher.get_v_target(x_t, t, encoder_hidden_states)

        # Student forward
        v_pred, s_features = student.forward_with_features(
            x_t, t, encoder_hidden_states
        )

        return {
            'teacher': (eps_pred, t_features),
            'student': (v_pred, s_features),
            'teacher_v_target': v_target
        }

    return forward


# =====================================================================================
# Main Training Setup
# =====================================================================================

def main():
    # Configuration
    cfg = SD15FlowConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Configuration:")
    print(json.dumps(asdict(cfg), indent=2))

    # Load SD1.5 pipeline
    print("\nLoading SD1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)

    # Create models
    print("Creating models...")

    # Teacher
    teacher = SD15Teacher(
        unet=pipe.unet,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        scheduler=scheduler,
        block_names=cfg.active_blocks,
        freeze=True
    )

    # Student (copy of teacher UNet)
    student_unet = pipe.unet.__class__.from_config(pipe.unet.config)
    student_unet.load_state_dict(pipe.unet.state_dict(), strict=True)
    student = SD15Student(
        unet=student_unet.to(device),
        block_names=cfg.active_blocks,
        use_local_flow_heads=cfg.use_local_flow_heads
    )

    # David (assistant)
    david_model = load_david(cfg, device)
    assistant = DavidAssistant(
        david_model=david_model,
        pooling=cfg.pooling,
        freeze=True
    )

    # Setup Arbitrator
    print("\nSetting up Arbitrator...")
    arb_cfg = ArbitratorConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        freeze_teacher=True,
        freeze_assistant=True
    )

    trainer_cfg = TrainerConfig(
        run_name=cfg.run_name,
        out_dir=cfg.out_dir,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        amp=cfg.amp
    )

    arbitrator = Arbitrator(arb_cfg, trainer_cfg, device=device)
    arbitrator.register_teacher(teacher)
    arbitrator.register_student(student)
    arbitrator.register_assistant(assistant)

    # Setup losses
    print("Configuring losses...")
    loss_composer = LossComposer()
    loss_composer.register("flow", create_flow_loss(cfg), cfg.global_flow_weight)
    loss_composer.register("kd", create_kd_loss(cfg), cfg.kd_weight)
    loss_composer.register("local_flow", create_local_flow_loss(cfg), 1.0)
    loss_composer.register("david_penalty", create_david_penalty_loss(cfg, assistant), cfg.block_penalty_weight)

    arbitrator.register_loss_composer(loss_composer)

    # Setup data
    print("Setting up data...")
    data_factory = create_data_factory(cfg)
    arbitrator.register_data_factory(data_factory)

    # Setup checkpointing
    print("Setting up checkpointing...")
    ckpt_cfg = CheckpointConfig(
        ckpt_dir=cfg.ckpt_dir,
        save_every=cfg.save_every,
        hf_repo_id=None,  # Set to upload to HF
        auto_generate_card=True
    )
    checkpointer = Checkpointer(ckpt_cfg)

    # Build trainer
    print("Building trainer...")
    trainer = arbitrator.build_trainer()

    # Create custom forward
    forward_fn = create_forward_fn(teacher, student, cfg)

    # Train!
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    arbitrator.train(custom_forward=forward_fn)

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    state = arbitrator.get_state_dict()
    config = asdict(cfg)
    checkpointer.save_local(state, config, tag="final")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()