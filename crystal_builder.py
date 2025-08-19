# crystal_lattice/crystal_builder.py
import numpy as np
from .warp_field import symbolic_warp_tensor, role_displacement_field
import math
from typing import Dict

DIM = 512
EPS = 1e-6

# ======================================================
# Cayley-Menger Volume Calculation
# ======================================================
def cayley_menger_volume(p: np.ndarray) -> float:
    from scipy.spatial.distance import pdist, squareform
    D2 = squareform(pdist(p, metric='euclidean')) ** 2
    M = np.ones((6, 6), dtype=np.float32)
    M[1:, 1:] = D2
    M[0, 0] = 0
    det = np.linalg.det(M)
    return math.sqrt(abs(det) / 288.0 + EPS)

# ======================================================
# Cardinal Assignment
# ======================================================
def assign_cardinal(word: str) -> str:
    if word.isupper(): return "ℵ₂"
    if word.endswith("ing") or "-" in word: return "ℵ₁"
    return "ℵ₀"

# ======================================================
# Final Crystal Assembly
# ======================================================
def generate_crystal(word: str) -> Dict:
    warp_tensor = symbolic_warp_tensor(word)  # [512]
    disp = role_displacement_field(word)      # [5, 512]
    crystal = disp + warp_tensor              # Warp-based geometry

    # Normalize
    crystal -= crystal.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(crystal, axis=1).mean()
    crystal /= max(scale, EPS)

    # Validate
    volume = cayley_menger_volume(crystal)

    return {
        "word": word,
        "cardinal": assign_cardinal(word),
        "crystal": crystal.astype(np.float32),
        "volume": float(volume),
    }