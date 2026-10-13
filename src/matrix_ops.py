from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocess import MinMaxScalerCustom, StandardScalerCustom


def create_sparse_array(seed: int = 42, density: float = 1e-5):
    """Разреженная матрица 100000 x 100000 без выделения плотного массива."""
    rng = np.random.default_rng(seed)
    s = sparse.random(
        100_000,
        100_000,
        density=density,
        format="csr",
        random_state=rng,
        data_rvs=rng.random,
    )
    return s


def inverse_random_matrix(seed: int = 42, size: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    # Добавляем диагональное усиление, чтобы матрица была хорошо обусловлена и обратима.
    matrix = rng.normal(size=(size, size)) + np.eye(size) * 0.5
    try:
        inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Случайная матрица оказалась вырожденной") from exc
    error = float(np.max(np.abs(matrix @ inv - np.eye(size))))
    return matrix, inv, error


def build_feature_tensor(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Формирует F=(n, m, channels): исходные, масштабы и 4 бинарные маски."""
    x = np.asarray(data, dtype=np.float64)
    mm01 = MinMaxScalerCustom(0.0, 1.0).fit_transform(x)
    mm_ab = MinMaxScalerCustom(-1.0, 1.0).fit_transform(x)
    zscore = StandardScalerCustom(ddof=0).fit_transform(x)

    positive = (x > 0).astype(np.float64)
    negative = (x < 0).astype(np.float64)
    zero = (x == 0).astype(np.float64)
    interval = ((x >= -1) & (x <= 1)).astype(np.float64)

    channels = [
        "original",
        "minmax_0_1",
        "minmax_-1_1",
        "zscore",
        "mask_positive",
        "mask_negative",
        "mask_zero",
        "mask_interval_-1_1",
    ]
    f = np.stack([x, mm01, mm_ab, zscore, positive, negative, zero, interval], axis=-1)
    return f, channels


def _feature_sets(f: np.ndarray) -> dict[str, np.ndarray]:
    # channel axis = 2; reshape сохраняет наблюдения по первой оси.
    def take(channels: list[int]) -> np.ndarray:
        return f[:, :, channels].reshape(len(f), -1)

    return {
        "original": take([0]),
        "scaled_representations": take([1, 2, 3]),
        "original_plus_masks": take([0, 4, 5, 6, 7]),
        "all_F_channels": take(list(range(f.shape[2]))),
    }


def pca_pipeline_search(f: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Ищет лучшую комбинацию набора признаков F и масштабирования для 2D PCA."""
    sets = _feature_sets(f)
    scalers = {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
    }

    rows: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    for set_name, x in sets.items():
        x_train, x_val = train_test_split(x, test_size=0.30, random_state=seed, shuffle=True)
        for scaler_name, scaler in scalers.items():
            n_components = min(2, x.shape[1])
            pipe = Pipeline([
                ("scaler", scaler),
                ("pca", PCA(n_components=n_components, svd_solver="full")),
            ])
            pipe.fit(x_train)
            x_val_scaled = pipe.named_steps["scaler"].transform(x_val)
            pca = pipe.named_steps["pca"]
            log_likelihood = float(pca.score(x_val_scaled))
            normalized_ll = log_likelihood / x.shape[1]
            explained = float(np.sum(pca.explained_variance_ratio_))
            row = {
                "feature_set": set_name,
                "scaler": scaler_name,
                "n_features": int(x.shape[1]),
                "n_components": int(n_components),
                "validation_log_likelihood": log_likelihood,
                "log_likelihood_per_feature": normalized_ll,
                "explained_variance_2d": explained,
            }
            rows.append(row)
            candidates[(set_name, scaler_name)] = {"x": x, "row": row}

    comparison = pd.DataFrame(rows).sort_values(
        ["log_likelihood_per_feature", "explained_variance_2d"], ascending=False
    ).reset_index(drop=True)
    best_row = comparison.iloc[0].to_dict()
    best_key = (str(best_row["feature_set"]), str(best_row["scaler"]))
    best_x = candidates[best_key]["x"]

    best_scaler = MinMaxScaler() if best_key[1] == "MinMaxScaler" else StandardScaler()
    best_scaled = best_scaler.fit_transform(best_x)
    best_pca = PCA(n_components=min(2, best_x.shape[1]), svd_solver="full").fit(best_scaled)
    projection = best_pca.transform(best_scaled)

    fig_compare, ax = plt.subplots(figsize=(11, 5))
    labels = comparison["feature_set"] + "\n" + comparison["scaler"]
    ax.bar(np.arange(len(comparison)), comparison["log_likelihood_per_feature"])
    ax.set_xticks(np.arange(len(comparison)), labels, rotation=30, ha="right")
    ax.set_ylabel("Средний log-likelihood на признак")
    ax.set_title("Сравнение комбинаций масштабирования и признаков F")
    ax.grid(True, axis="y", alpha=0.25)
    fig_compare.tight_layout()

    fig_proj, axp = plt.subplots(figsize=(8, 6))
    if projection.shape[1] >= 2:
        scatter = axp.scatter(projection[:, 0], projection[:, 1], c=np.arange(len(projection)), s=18)
        fig_proj.colorbar(scatter, ax=axp, label="Индекс наблюдения")
        axp.set_ylabel("PC2")
    else:
        axp.scatter(projection[:, 0], np.zeros(len(projection)), c=np.arange(len(projection)), s=18)
        axp.set_ylabel("0")
    axp.set_xlabel("PC1")
    axp.set_title(f"Лучшая PCA-проекция: {best_key[0]} + {best_key[1]}")
    axp.grid(True, alpha=0.25)
    fig_proj.tight_layout()

    # Логарифм правдоподобия в зависимости от числа компонент для лучшей комбинации.
    x_train, x_val = train_test_split(best_x, test_size=0.30, random_state=seed, shuffle=True)
    scaler_for_curve = MinMaxScaler() if best_key[1] == "MinMaxScaler" else StandardScaler()
    x_train_s = scaler_for_curve.fit_transform(x_train)
    x_val_s = scaler_for_curve.transform(x_val)
    max_components = min(10, x_train_s.shape[1] - 1, x_train_s.shape[0] - 1)
    max_components = max(1, max_components)
    comps = np.arange(1, max_components + 1)
    ll_curve = []
    for n_comp in comps:
        pca = PCA(n_components=int(n_comp), svd_solver="full").fit(x_train_s)
        ll_curve.append(float(pca.score(x_val_s)))

    fig_ll, axll = plt.subplots(figsize=(9, 5))
    axll.plot(comps, ll_curve, marker="o")
    axll.set_xlabel("Число главных компонент")
    axll.set_ylabel("Log-likelihood на validation")
    axll.set_title("Логарифмический график правдоподобия PCA")
    axll.grid(True, alpha=0.3)
    fig_ll.tight_layout()

    return {
        "comparison": comparison,
        "best": best_row,
        "projection": projection,
        "fig_comparison": fig_compare,
        "fig_projection": fig_proj,
        "fig_log_likelihood": fig_ll,
        "log_likelihood_components": comps,
        "log_likelihood_values": np.asarray(ll_curve),
    }
