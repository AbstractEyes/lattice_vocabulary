"""
Checkpointing system with HuggingFace Hub integration.
Handles saving/loading, config management, and model card generation.
"""
from __future__ import annotations
import json
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, snapshot_download, create_repo


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    ckpt_dir: str = "./checkpoints"
    save_every: int = 1  # epochs
    save_format: str = "safetensors"  # or "pytorch"

    # HuggingFace
    hf_repo_id: Optional[str] = None
    hf_private: bool = False
    hf_commit_message: str = "Update checkpoint"

    # Model card
    auto_generate_card: bool = True
    model_card_template: Optional[str] = None

    def __post_init__(self):
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)


class Checkpointer:
    """
    Manages model checkpointing and HuggingFace Hub integration.
    """
    def __init__(self, cfg: CheckpointConfig):
        self.cfg = cfg
        self.api = HfApi() if cfg.hf_repo_id else None

    def save_local(
        self,
        state_dict: Dict[str, Any],
        config: Dict[str, Any],
        tag: str = "checkpoint",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint locally.

        Args:
            state_dict: Model state dictionary
            config: Training configuration
            tag: Checkpoint identifier (epoch number, "final", etc.)
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint directory
        """
        ckpt_path = Path(self.cfg.ckpt_dir) / f"{tag}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        if self.cfg.save_format == "safetensors":
            # Filter to only tensor values for safetensors
            tensor_state = {k: v for k, v in state_dict.items()
                           if isinstance(v, torch.Tensor)}
            weights_path = ckpt_path / "model.safetensors"
            save_file(tensor_state, str(weights_path))
        else:  # pytorch
            weights_path = ckpt_path / "pytorch_model.bin"
            torch.save(state_dict, weights_path)

        # Save config
        config_path = ckpt_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save metadata if provided
        if metadata:
            meta_path = ckpt_path / "training_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Generate model card if enabled
        if self.cfg.auto_generate_card:
            self._generate_model_card(ckpt_path, config, metadata)

        print(f"✓ Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def load_local(self, tag: str = "checkpoint") -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint from local path.

        Returns:
            (state_dict, config)
        """
        ckpt_path = Path(self.cfg.ckpt_dir) / f"{tag}"

        # Load weights
        if (ckpt_path / "model.safetensors").exists():
            state_dict = load_file(str(ckpt_path / "model.safetensors"))
        elif (ckpt_path / "pytorch_model.bin").exists():
            state_dict = torch.load(ckpt_path / "pytorch_model.bin")
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_path}")

        # Load config
        config_path = ckpt_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"✓ Checkpoint loaded: {ckpt_path}")
        return state_dict, config

    def save_hf(
        self,
        state_dict: Dict[str, Any],
        config: Dict[str, Any],
        commit_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save checkpoint to HuggingFace Hub.
        """
        if not self.cfg.hf_repo_id:
            raise ValueError("hf_repo_id not configured")

        # First save locally
        tag = "hf_upload"
        ckpt_path = self.save_local(state_dict, config, tag, metadata)

        # Create repo if it doesn't exist
        try:
            create_repo(
                self.cfg.hf_repo_id,
                private=self.cfg.hf_private,
                exist_ok=True
            )
        except Exception as e:
            print(f"Note: {e}")

        # Upload
        message = commit_message or self.cfg.hf_commit_message
        self.api.upload_folder(
            repo_id=self.cfg.hf_repo_id,
            folder_path=str(ckpt_path),
            commit_message=message
        )

        print(f"✓ Uploaded to HuggingFace: {self.cfg.hf_repo_id}")

    def load_hf(
        self,
        repo_id: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint from HuggingFace Hub.

        Returns:
            (state_dict, config)
        """
        repo_id = repo_id or self.cfg.hf_repo_id
        if not repo_id:
            raise ValueError("repo_id must be provided or configured")

        cache_dir = cache_dir or "./_hf_cache"

        # Download repo
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )

        # Load weights
        if (Path(local_path) / "model.safetensors").exists():
            state_dict = load_file(str(Path(local_path) / "model.safetensors"))
        elif (Path(local_path) / "pytorch_model.bin").exists():
            state_dict = torch.load(Path(local_path) / "pytorch_model.bin")
        else:
            raise FileNotFoundError(f"No model weights found in {local_path}")

        # Load config
        with open(Path(local_path) / "config.json", 'r') as f:
            config = json.load(f)

        print(f"✓ Loaded from HuggingFace: {repo_id}")
        return state_dict, config

    def _generate_model_card(
        self,
        ckpt_path: Path,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Generate a basic model card."""
        if self.cfg.model_card_template:
            # Use custom template
            with open(self.cfg.model_card_template, 'r') as f:
                card_content = f.read()
        else:
            # Generate basic card
            card_content = self._default_model_card(config, metadata)

        card_path = ckpt_path / "README.md"
        with open(card_path, 'w') as f:
            f.write(card_content)

    def _default_model_card(
        self,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate default model card content."""
        card = f"""# Model Card

## Model Details

Training configuration:
```json
{json.dumps(config, indent=2)}
```

"""

        if metadata:
            card += f"""## Training Metadata

```json
{json.dumps(metadata, indent=2)}
```

"""

        card += """## Usage

```python
from safetensors.torch import load_file

# Load weights
state_dict = load_file("model.safetensors")

# Load config
import json
with open("config.json", "r") as f:
    config = json.load(f)
```

## Citation

```bibtex
@software{model2024,
  author = {Your Name},
  title = {Model Name},
  year = {2024}
}
```
"""

        return card


class CheckpointCallback:
    """
    Callback for periodic checkpoint saving during training.
    Can be integrated with Trainer or Arbitrator.
    """
    def __init__(
        self,
        checkpointer: Checkpointer,
        get_state_fn: callable,
        get_config_fn: callable,
        save_every: int = 1,
        save_to_hf: bool = False
    ):
        self.checkpointer = checkpointer
        self.get_state_fn = get_state_fn
        self.get_config_fn = get_config_fn
        self.save_every = save_every
        self.save_to_hf = save_to_hf

    def on_epoch_end(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Called at end of each epoch."""
        if (epoch + 1) % self.save_every == 0:
            state_dict = self.get_state_fn()
            config = self.get_config_fn()

            metadata = {
                'epoch': epoch + 1,
                'metrics': metrics or {}
            }

            # Save locally
            self.checkpointer.save_local(
                state_dict=state_dict,
                config=config,
                tag=f"epoch_{epoch+1}",
                metadata=metadata
            )

            # Optionally upload to HF
            if self.save_to_hf:
                self.checkpointer.save_hf(
                    state_dict=state_dict,
                    config=config,
                    commit_message=f"Epoch {epoch+1}",
                    metadata=metadata
                )

    def on_training_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """Called at end of training."""
        state_dict = self.get_state_fn()
        config = self.get_config_fn()

        metadata = {
            'status': 'training_complete',
            'final_metrics': final_metrics or {}
        }

        # Save final checkpoint
        self.checkpointer.save_local(
            state_dict=state_dict,
            config=config,
            tag="final",
            metadata=metadata
        )

        # Upload final to HF
        if self.save_to_hf:
            self.checkpointer.save_hf(
                state_dict=state_dict,
                config=config,
                commit_message="Training complete",
                metadata=metadata
            )