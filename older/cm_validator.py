# crystal_lattice/cm_validator.py
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform

EPS = 1e-6

# ======================================================
# Standalone Cayley-Menger Volume Calculation
# ======================================================
def cayley_menger_volume(p: np.ndarray) -> float:
    if p.shape != (5, 512):
        raise ValueError("Crystal must be shape [5, 512]")

    D2 = squareform(pdist(p, metric='euclidean')) ** 2
    M = np.ones((6, 6), dtype=np.float32)
    M[1:, 1:] = D2
    M[0, 0] = 0.0
    det = np.linalg.det(M)
    volume_sq = max(det / 288.0, 0.0)
    return math.sqrt(volume_sq + EPS)

# ======================================================
# Validity Threshold
# ======================================================
def is_valid_crystal(p: np.ndarray, volume_thresh: float = 0.001) -> bool:
    vol = cayley_menger_volume(p)
    return vol > volume_thresh

# ======================================================
# Test Utility
# ======================================================
def validate_batch(crystals: dict, verbose=False) -> dict:
    results = {}
    for word, crystal in crystals.items():
        vol = cayley_menger_volume(crystal)
        results[word] = vol
        if verbose and vol < 0.001:
            print(f"[warn] degenerate crystal: {word} → vol={vol:.6f}")
    return results
