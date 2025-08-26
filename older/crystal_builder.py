# crystal_lattice/crystal_builder.py (patch snippet)
import numpy as np
from typing import Dict
from .specials import SPECIAL_TOKEN_ORDER, RESERVED_BAND_SIZE
from .special_crystals import special_crystal_for
from .warp_field import symbolic_warp_tensor, role_displacement_field
from .cm_validator import cayley_menger_volume

DIM = 512
EPS = 1e-6
_SPECIAL_SET = set(SPECIAL_TOKEN_ORDER)

def _assign_cardinal(word: str) -> str:
    # You can replace with your richer rules if desired
    if word.isupper(): return "ℵ₂"
    if word.endswith("ing") or "-" in word: return "ℵ₁"
    return "ℵ₀"

def generate_crystal(token: str) -> Dict:
    if token in _SPECIAL_SET:
        crystal = special_crystal_for(token, dim=DIM)
        volume = float(cayley_menger_volume(crystal))
        return {
            "word": token,
            "cardinal": "ℵ₀",  # specials live outside ℵ; keep placeholder
            "crystal": crystal.astype(np.float32),
            "volume": volume,
            "is_special": True,
        }

    # Normal word path: warp + role displacement → normalized
    warp_tensor = symbolic_warp_tensor(token)     # [512]
    disp = role_displacement_field(token)         # [5, 512]
    crystal = disp + warp_tensor                  # [5, 512]
    crystal -= crystal.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(crystal, axis=1).mean()
    crystal /= max(scale, EPS)
    volume = float(cayley_menger_volume(crystal))
    return {
        "word": token,
        "cardinal": _assign_cardinal(token),
        "crystal": crystal.astype(np.float32),
        "volume": volume,
        "is_special": False,
    }
