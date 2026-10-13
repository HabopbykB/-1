from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocess import MinMaxScalerCustom, StandardScalerCustom


def compare_scalers(data: np.ndarray, columns: list[str]) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    data = np.asarray(data, dtype=np.float64)

    custom_mm = MinMaxScalerCustom(0.0, 1.0).fit(data)
    custom_ab = MinMaxScalerCustom(-1.0, 1.0).fit(data)
    custom_std = StandardScalerCustom(ddof=0).fit(data)

    sk_mm = MinMaxScaler(feature_range=(0.0, 1.0)).fit(data)
    sk_ab = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(data)
    sk_std = StandardScaler().fit(data)

    arrays = {
        "original": data,
        "custom_minmax_01": custom_mm.transform(data),
        "sklearn_minmax_01": sk_mm.transform(data),
        "custom_minmax_-1_1": custom_ab.transform(data),
        "sklearn_minmax_-1_1": sk_ab.transform(data),
        "custom_standard": custom_std.transform(data),
        "sklearn_standard": sk_std.transform(data),
    }

    assert np.allclose(arrays["custom_minmax_01"], arrays["sklearn_minmax_01"], atol=1e-10)
    assert np.allclose(arrays["custom_minmax_-1_1"], arrays["sklearn_minmax_-1_1"], atol=1e-10)
    assert np.allclose(arrays["custom_standard"], arrays["sklearn_standard"], atol=1e-10)

    flat = {}
    for key, arr in arrays.items():
        for i, col in enumerate(columns):
            flat[f"{key}__{col}"] = arr[:, i]
    result_df = pd.DataFrame(flat)

    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4 * len(columns)), squeeze=False)
    for i, col in enumerate(columns):
        ax = axes[i, 0]
        ax.plot(arrays["custom_minmax_01"][:, i], label="custom MinMax [0,1]")
        ax.plot(arrays["sklearn_minmax_01"][:, i], linestyle="--", label="sklearn MinMax [0,1]")
        ax.plot(arrays["custom_standard"][:, i], alpha=0.75, label="custom z-score")
        ax.plot(arrays["sklearn_standard"][:, i], linestyle=":", label="sklearn StandardScaler")
        ax.set_title(f"Сравнение масштабирования: {col}")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()

    inverses = {
        "custom_mm_error": float(np.max(np.abs(custom_mm.inverse_transform(arrays["custom_minmax_01"]) - data))),
        "sklearn_mm_error": float(np.max(np.abs(sk_mm.inverse_transform(arrays["sklearn_minmax_01"]) - data))),
        "custom_ab_error": float(np.max(np.abs(custom_ab.inverse_transform(arrays["custom_minmax_-1_1"]) - data))),
        "sklearn_ab_error": float(np.max(np.abs(sk_ab.inverse_transform(arrays["sklearn_minmax_-1_1"]) - data))),
        "custom_std_error": float(np.max(np.abs(custom_std.inverse_transform(arrays["custom_standard"]) - data))),
        "sklearn_std_error": float(np.max(np.abs(sk_std.inverse_transform(arrays["sklearn_standard"]) - data))),
    }

    details = {
        "arrays": arrays,
        "custom_minmax": custom_mm,
        "custom_ab": custom_ab,
        "custom_standard": custom_std,
        "sklearn_minmax": sk_mm,
        "sklearn_ab": sk_ab,
        "sklearn_standard": sk_std,
        "inverse_errors": inverses,
    }
    return result_df, fig, details
