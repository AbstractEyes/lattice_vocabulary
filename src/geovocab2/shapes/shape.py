"""
    Geometric Shapes Module
    -------------------------
    Author: AbstractPhil
    -------------------------
    Description: This module defines various geometric shapes and their properties.
    They are specifically meant to be optimized for rapid data moving between hardware variants.

    License: Apache License 2.0

    Currently unused but will have a wrappered hierarchy when the system is more stable and tested.
"""
import numpy as np


# --- Core Shape Generators (emit NumPy) ---
def regular_simplex(dim: int, dtype=np.float32) -> np.ndarray:
    E = np.eye(dim, dtype=np.float64)
    centroid = np.mean(E, axis=0, keepdims=True)
    S = E - centroid
    edge_length = np.linalg.norm(S[0] - S[1])
    S = S / edge_length
    return S.astype(dtype)


def pentachoron(dtype=np.float32) -> np.ndarray:
    return regular_simplex(5, dtype=dtype)


def orthoplex(dim: int, dtype=np.float32) -> np.ndarray:
    verts = []
    for i in range(dim):
        e = np.zeros(dim, dtype=np.float64)
        e[i] = 1.0
        verts.append(e)
        verts.append(-e)
    return np.array(verts, dtype=dtype)


def hypercube(dim: int, dtype=np.float32) -> np.ndarray:
    verts = []
    for i in range(2 ** dim):
        bits = [(1 if (i >> j) & 1 else -1) for j in range(dim)]
        verts.append(bits)
    return np.array(verts, dtype=dtype)

