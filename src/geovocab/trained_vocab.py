"""
    Trained Vocab
    Author: AbstractPhil
    License : MIT

    This module provides simplified use-case examples to efficiently store and load
    a vocabulary of geometric tokens used in models, focusing on minimizing storage
    by only keeping track of tokens that were actually used and how they were
    transformed from a base pretrained vocabulary.


"""

import numpy as np
from datasets import load_dataset
from .pretrained_geometric_vocab import PretrainedGeometricVocab


class UsedVocabSnapshot:
    """Stores only the manifestation trail for used tokens."""

    def __init__(self, base_vocab: PretrainedGeometricVocab):
        self.base_repo = base_vocab.repo_id
        self.base_split = "wordnet_eng"  # or from config
        self.dim = base_vocab.dim

        # Track only what changed from base
        self.manifested_tokens = {}  # token_id -> manifestation_record
        self.role_mappings = {}  # how tokens map to model roles

    def record_token_use(self, token: str, token_id: int,
                         crystal: np.ndarray, role: str = "anchor"):
        """Record that this token was used, possibly transformed."""

        base_crystal = self.base_vocab.embedding(token_id)

        # Only store if modified from base
        if base_crystal is None or not np.allclose(crystal, base_crystal):
            # Store the transformation, not the full crystal
            self.manifested_tokens[token_id] = {
                "token": token,
                "transform": self._compute_transform(base_crystal, crystal),
                "role": role
            }
        else:
            # Just mark as used with role
            self.role_mappings[token_id] = role

    def _compute_transform(self, base: np.ndarray, manifested: np.ndarray):
        """Store minimal transform info."""
        if base is None:
            return {"type": "generated", "pooled": manifested.mean(0)}

        # Could store as delta, scale factor, rotation, etc.
        delta = manifested - base
        if np.abs(delta).max() < 1e-6:
            return {"type": "identity"}

        # Compress delta with PCA or just store pooled
        return {
            "type": "delta_pooled",
            "value": delta.mean(0).astype(np.float16)  # reduced precision
        }


def export_used_vocab(snapshot: UsedVocabSnapshot) -> dict:
    """Export to minimal dataset format."""

    # Sort tokens by frequency of use (if tracked)
    used_ids = sorted(set(snapshot.manifested_tokens.keys()) |
                      set(snapshot.role_mappings.keys()))

    records = []
    for tid in used_ids:
        if tid in snapshot.manifested_tokens:
            rec = snapshot.manifested_tokens[tid]
            records.append({
                "token_id": tid,
                "token": rec["token"],
                "transform_type": rec["transform"]["type"],
                "transform_data": rec["transform"].get("value"),
                "role": rec["role"]
            })
        else:
            records.append({
                "token_id": tid,
                "role": snapshot.role_mappings[tid]
            })

    return {
        "base_vocab": {
            "repo": snapshot.base_repo,
            "split": snapshot.base_split,
            "dim": snapshot.dim
        },
        "used_tokens": records,
        "stats": {
            "total_used": len(used_ids),
            "manifested": len(snapshot.manifested_tokens),
            "base_only": len(snapshot.role_mappings)
        }
    }


class EfficientVocabLoader:
    """Reconstruct vocabulary from snapshot."""

    def __init__(self, snapshot_path: str):
        data = load_dataset(snapshot_path)

        # Load base vocab (cached across models using same base)
        self.base = PretrainedGeometricVocab(
            repo_id=data["base_vocab"]["repo"],
            split=data["base_vocab"]["split"],
            dim=data["base_vocab"]["dim"],
            manifest_specials=False
        )

        # Apply transforms lazily
        self.transforms = {
            rec["token_id"]: rec
            for rec in data["used_tokens"]
            if "transform_type" in rec
        }

    def get_crystal(self, token_id: int) -> np.ndarray:
        if token_id in self.transforms:
            base = self.base.embedding(token_id)
            return self._apply_transform(base, self.transforms[token_id])
        return self.base.embedding(token_id)