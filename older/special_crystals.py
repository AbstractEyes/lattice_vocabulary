# crystal_lattice/special_crystals.py
import numpy as np
import hashlib

DIM = 512
EPS = 1e-6

def _canon_basis(dim: int = DIM) -> np.ndarray:
    """
    Deterministic 5xD orthonormal-like basis from fixed primes.
    We don't rely on QR of random matrices for specials to ensure invariance.
    """
    # Use a simple frequency grid with 5 unique phases
    phases = np.linspace(0.0, np.pi, 5, endpoint=False)
    base = np.linspace(-1.0, 1.0, dim)
    B = []
    for p in phases:
        v = np.sin(base * (1.0 + base**2) + p)
        v /= (np.linalg.norm(v) + EPS)
        B.append(v)
    B = np.stack(B, axis=0)   # [5, D]
    # Gram-Schmidt light orthogonalization pass
    for i in range(5):
        for j in range(i):
            proj = np.dot(B[i], B[j]) * B[j]
            B[i] = B[i] - proj
        B[i] /= (np.linalg.norm(B[i]) + EPS)
    return B

_CANON = _canon_basis()

def special_crystal_for(name: str, dim: int = DIM) -> np.ndarray:
    """
    Canonical, CM-valid special-token crystal:
    - no hash randomness
    - centered, normalized
    """
    # Stable per-token perturbation from name
    h = hashlib.sha256(name.encode("utf-8")).digest()
    jitter = (np.frombuffer(h, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) / 127.5
    jitter *= 0.02  # very small perturbation

    X = _CANON + jitter[None, :]
    X -= X.mean(axis=0, keepdims=True)
    r = np.sqrt((X * X).sum(axis=1).mean())
    X = X / max(r, EPS)
    return X.astype(np.float32)
