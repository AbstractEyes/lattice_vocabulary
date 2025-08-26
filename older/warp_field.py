# crystal_lattice/warp_field.py
import numpy as np
import hashlib

DIM = 512
EPS = 1e-6


def symbolic_warp_tensor(word: str) -> np.ndarray:
    h = hashlib.sha256(word.encode()).digest()
    primes = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:DIM]
    primes = 1 + (primes % 11)
    base = np.linspace(-1.0, 1.0, DIM)
    curve = np.sin(base * primes)
    tensor = curve * np.tanh(base * np.sum(primes) / DIM)
    return tensor / (np.linalg.norm(tensor) + EPS)


def role_displacement_field(word: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    base_field = rng.normal(0, 1, (5, DIM))
    warp = symbolic_warp_tensor(word)
    warped = base_field + 0.1 * (warp[None, :] * base_field)
    warped -= warped.mean(axis=0, keepdims=True)
    return warped / (np.linalg.norm(warped, axis=1, keepdims=True) + EPS)
