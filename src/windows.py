from __future__ import annotations

import numpy as np


def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Функция ожидает одномерный массив")
    if width <= 0 or width > len(x):
        raise ValueError("Некорректная ширина окна")
    return np.lib.stride_tricks.sliding_window_view(x, window_shape=width)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    return sliding_window(x, width).mean(axis=1)
