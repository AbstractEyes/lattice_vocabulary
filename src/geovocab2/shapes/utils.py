import numpy as np


try:
    import torch
    HAS_TORCH = True
except ImportError:

    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def torch_available() -> bool:
    return HAS_TORCH

def tf_available() -> bool:
    return HAS_TF


# --- Converters ---
def to_numpy(array: np.ndarray, dtype=np.float32) -> np.ndarray:
    return np.asarray(array, dtype=dtype)


def to_torch(array: np.ndarray, device="cpu", dtype=None):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed")
    return torch.from_numpy(np.asarray(array)).to(device=device, dtype=dtype or torch.float32)


def to_tensorflow(array: np.ndarray, device="/CPU:0", dtype=None):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed")
    with tf.device(device):
        return tf.convert_to_tensor(np.asarray(array), dtype=dtype or tf.float32)