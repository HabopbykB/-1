from __future__ import annotations

from typing import Any

import numpy as np


def make_tensorflow_tensor(data: np.ndarray):
    import tensorflow as tf
    return tf.constant(data, dtype=tf.float32)


def make_pytorch_tensor(data: np.ndarray):
    import torch
    return torch.tensor(data, dtype=torch.float32)


def tensorflow_axwb(data: np.ndarray, k: int = 5, p: int = 4, seed: int = 42) -> dict[str, Any]:
    import tensorflow as tf

    tf.random.set_seed(seed)
    x = tf.constant(data, dtype=tf.float32)
    n, m = data.shape
    a = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32, seed=seed)
    w = tf.random.normal((m, p), seed=seed + 1)
    b = tf.random.uniform((k, p), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed + 2)
    result = tf.matmul(tf.matmul(tf.cast(a, tf.float32), x), w) + b
    return {"X": x, "A": a, "W": w, "B": b, "result": result}


def pytorch_axwb(data: np.ndarray, k: int = 5, p: int = 4, seed: int = 42) -> dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.tensor(data, dtype=torch.float32)
    n, m = data.shape
    a = torch.randint(0, 10, (k, n), generator=generator, dtype=torch.int64).to(torch.float32)
    w = torch.randn((m, p), generator=generator, dtype=torch.float32)
    b = torch.rand((k, p), generator=generator, dtype=torch.float32)
    result = a @ x @ w + b
    return {"X": x, "A": a, "W": w, "B": b, "result": result}


def create_tpq(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Диапазоны различаются, поэтому матрицы гарантированно отличаются.
    t = rng.uniform(0.05, 0.95, size=(3, 10))
    p = rng.uniform(0.10, 0.80, size=(3, 10))
    q = rng.uniform(0.20, 1.20, size=(3, 10))  # Q > 0 для sqrt
    return t, p, q


def keras_operation(t: np.ndarray, p: np.ndarray, q: np.ndarray):
    from keras import ops

    t_k = ops.convert_to_tensor(t)
    p_k = ops.convert_to_tensor(p)
    q_k = ops.convert_to_tensor(q)
    return ops.abs(ops.sin(t_k) - ops.exp(p_k) * ops.sqrt(q_k))


def pytorch_operation(t: np.ndarray, p: np.ndarray, q: np.ndarray):
    import torch

    t_t = torch.tensor(t, dtype=torch.float64)
    p_t = torch.tensor(p, dtype=torch.float64)
    q_t = torch.tensor(q, dtype=torch.float64)
    return torch.abs(torch.sin(t_t) - torch.exp(p_t) * torch.sqrt(q_t))
