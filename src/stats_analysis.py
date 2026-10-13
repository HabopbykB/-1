from __future__ import annotations

from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import periodogram, spectrogram


def plot_series(data: np.ndarray, columns: list[str]):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(columns):
        ax.plot(data[:, i], label=col, linewidth=1.2)
    ax.set_title("Исходные данные")
    ax.set_xlabel("Индекс наблюдения")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histograms(data: np.ndarray, columns: list[str], bins: int = 30):
    ncols = len(columns)
    fig, axes = plt.subplots(ncols, 1, figsize=(10, 3.5 * ncols), squeeze=False)
    for i, col in enumerate(columns):
        ax = axes[i, 0]
        ax.hist(data[:, i], bins=bins, density=True, alpha=0.7, edgecolor="black")
        ax.set_title(f"{col}: нормализованная гистограмма")
        ax.set_xlabel("Значение")
        ax.set_ylabel("Плотность")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def sorted_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {col: np.sort(df[col].to_numpy(dtype=np.float64)) for col in df.columns}


def plot_ecdfs(data: np.ndarray, columns: list[str]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, col in enumerate(columns):
        xs = np.sort(data[:, i])
        ys = np.arange(1, len(xs) + 1, dtype=np.float64) / len(xs)
        ax.step(xs, ys, where="post", label=col)
    ax.set_title("Эмпирические функции распределения")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _mode_numpy(x: np.ndarray) -> float:
    """Мода только средствами NumPy: значение с максимальной частотой."""
    values, counts = np.unique(x, return_counts=True)
    return float(values[np.argmax(counts)])


def statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Статистики двумя способами: NumPy/SciPy и Pandas."""
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        pd_mode = df[col].mode(dropna=True)
        rows.append({
            "column": col,
            "numpy_mean": float(np.mean(x)),
            "pandas_mean": float(df[col].mean()),
            "numpy_var_sample": float(np.var(x, ddof=1)),
            "pandas_var_sample": float(df[col].var(ddof=1)),
            "scipy_mode": _mode_numpy(x),
            "pandas_mode": float(pd_mode.iloc[0]) if not pd_mode.empty else np.nan,
            "numpy_median": float(np.median(x)),
            "pandas_median": float(df[col].median()),
        })
    return pd.DataFrame(rows).set_index("column")


def plot_statistics(stats_df: pd.DataFrame):
    metrics = [
        ("numpy_mean", "Среднее"),
        ("numpy_var_sample", "Выборочная дисперсия"),
        ("scipy_mode", "Мода"),
        ("numpy_median", "Медиана"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (field, title) in zip(axes.ravel(), metrics):
        stats_df[field].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Столбец")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def confidence_intervals(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        n = len(x)
        mean = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        mean_ci = stats.t.interval(1 - alpha, df=n - 1, loc=mean, scale=s / np.sqrt(n))
        s2 = float(np.var(x, ddof=1))
        chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
        chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
        var_ci = ((n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low)
        rows.append({
            "column": col,
            "mean": mean,
            "mean_ci_low": float(mean_ci[0]),
            "mean_ci_high": float(mean_ci[1]),
            "variance": s2,
            "var_ci_low": float(var_ci[0]),
            "var_ci_high": float(var_ci[1]),
            "alpha": alpha,
        })
    return pd.DataFrame(rows).set_index("column")


def covariance_correlation(df: pd.DataFrame) -> dict[str, pd.DataFrame | np.ndarray]:
    data = df.to_numpy(dtype=np.float64)
    return {
        "numpy_cov": np.cov(data, rowvar=False, ddof=1),
        "numpy_corr": np.corrcoef(data, rowvar=False),
        "pandas_cov": df.cov(),
        "pandas_corr": df.corr(),
    }


def correlation_significance(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for a, b in combinations(df.columns, 2):
        r, p = stats.pearsonr(df[a].to_numpy(), df[b].to_numpy())
        rows.append({
            "column_1": a,
            "column_2": b,
            "pearson_r": float(r),
            "p_value": float(p),
            "alpha": alpha,
            "significant": bool(p < alpha),
        })
    return pd.DataFrame(rows)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lags: int = 50) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    max_lags = int(min(max_lags, len(x) - 1, len(y) - 1))
    x0 = x - x.mean()
    y0 = y - y.mean()
    full = np.correlate(x0, y0, mode="full")
    denom = np.std(x0) * np.std(y0) * min(len(x), len(y))
    if np.isclose(denom, 0):
        normalized = np.zeros_like(full, dtype=np.float64)
    else:
        normalized = full / denom
    lags_full = np.arange(-len(y) + 1, len(x))
    mask = (lags_full >= -max_lags) & (lags_full <= max_lags)
    return lags_full[mask], normalized[mask]


def plot_cross_correlations(df: pd.DataFrame, max_lags: int = 50):
    pairs = list(combinations(df.columns, 2))
    fig, axes = plt.subplots(max(1, len(pairs)), 1, figsize=(11, 3.2 * max(1, len(pairs))), squeeze=False)
    results: dict[str, dict[str, np.ndarray]] = {}
    if not pairs:
        axes[0, 0].text(0.5, 0.5, "Недостаточно столбцов", ha="center")
    for idx, (a, b) in enumerate(pairs):
        lags, corr = cross_correlation(df[a].to_numpy(), df[b].to_numpy(), max_lags=max_lags)
        results[f"{a}__{b}"] = {"lags": lags, "correlation": corr}
        ax = axes[idx, 0]
        ax.plot(lags, corr)
        ax.axvline(0, linewidth=1, linestyle="--")
        ax.set_title(f"Взаимная корреляция: {a} vs {b}")
        ax.set_xlabel("Сдвиг")
        ax.set_ylabel("Корреляция")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, results


def derivatives_and_gradients(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deriv = {}
    grad = {}
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        # Первая разность — численная производная при шаге 1; длину сохраняем.
        deriv[col] = np.diff(x, prepend=x[0])
        grad[col] = np.gradient(x)
    return pd.DataFrame(deriv), pd.DataFrame(grad)


def plot_derivatives(deriv: pd.DataFrame, grad: pd.DataFrame):
    fig, axes = plt.subplots(len(deriv.columns), 1, figsize=(12, 3.5 * len(deriv.columns)), squeeze=False)
    for i, col in enumerate(deriv.columns):
        ax = axes[i, 0]
        ax.plot(deriv[col].to_numpy(), label="разность")
        ax.plot(grad[col].to_numpy(), label="np.gradient", alpha=0.8)
        ax.set_title(f"Производная и градиент: {col}")
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def pairwise_convolutions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {}
    for a, b in combinations(df.columns, 2):
        result[f"{a}__{b}"] = np.convolve(df[a].to_numpy(), df[b].to_numpy(), mode="full")
    return result


def scalar_and_vector_products(df: pd.DataFrame) -> dict[str, Any]:
    if df.shape[1] < 2:
        raise ValueError("Нужно минимум два столбца")
    a, b = df.columns[:2]
    x = df[a].to_numpy(dtype=np.float64)
    y = df[b].to_numpy(dtype=np.float64)
    return {
        "columns": [a, b],
        "dot": float(np.dot(x, y)),
        "cross_first_3": np.cross(x[:3], y[:3]),
    }


def norms(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        rows.append({"column": col, "L1": float(np.linalg.norm(x, ord=1)), "L2": float(np.linalg.norm(x, ord=2))})
    return pd.DataFrame(rows).set_index("column")


def distribution_hypotheses(df: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    if df.shape[1] < 2:
        raise ValueError("Для задания 27 нужно минимум два столбца")
    first, second = df.columns[:2]
    x1 = df[first].to_numpy(dtype=np.float64)
    x2 = df[second].to_numpy(dtype=np.float64)

    loc = float(x1.min())
    scale = float(x1.max() - x1.min())
    uniform_stat, uniform_p = stats.kstest(x1, "uniform", args=(loc, scale))

    mean2 = float(x2.mean())
    std2 = float(x2.std(ddof=0))
    normal_stat, normal_p = stats.kstest(x2, "norm", args=(mean2, std2))
    shapiro_stat, shapiro_p = stats.shapiro(x2)

    return {
        "alpha": alpha,
        "uniform_first_column": {
            "column": first,
            "test": "Kolmogorov-Smirnov",
            "statistic": float(uniform_stat),
            "p_value": float(uniform_p),
            "do_not_reject_H0": bool(uniform_p > alpha),
        },
        "normal_second_column": {
            "column": second,
            "test": "Kolmogorov-Smirnov",
            "statistic": float(normal_stat),
            "p_value": float(normal_p),
            "do_not_reject_H0": bool(normal_p > alpha),
        },
        "normal_second_column_shapiro": {
            "column": second,
            "test": "Shapiro-Wilk",
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p),
            "do_not_reject_H0": bool(shapiro_p > alpha),
        },
    }


def plot_spectrograms(df: pd.DataFrame, fs: float = 1.0):
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(11, 3.8 * len(df.columns)), squeeze=False)
    for i, col in enumerate(df.columns):
        x = df[col].to_numpy(dtype=np.float64)
        f, t, sxx = spectrogram(x, fs=fs)
        ax = axes[i, 0]
        mesh = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-12), shading="auto")
        ax.set_title(f"Спектрограмма: {col}")
        ax.set_xlabel("Время / индекс")
        ax.set_ylabel("Частота")
        fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def plot_periodograms(df: pd.DataFrame, fs: float = 1.0):
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(11, 3.5 * len(df.columns)), squeeze=False)
    values = {}
    for i, col in enumerate(df.columns):
        f, pxx = periodogram(df[col].to_numpy(dtype=np.float64), fs=fs)
        values[col] = {"frequency": f, "psd": pxx}
        ax = axes[i, 0]
        ax.semilogy(f, pxx + 1e-20)
        ax.set_title(f"Периодограмма: {col}")
        ax.set_xlabel("Частота")
        ax.set_ylabel("PSD")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, values


def plot_fft_amplitudes(df: pd.DataFrame, sample_spacing: float = 1.0):
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(11, 3.5 * len(df.columns)), squeeze=False)
    values = {}
    for i, col in enumerate(df.columns):
        x = df[col].to_numpy(dtype=np.float64)
        n = len(x)
        fft_values = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, d=sample_spacing)
        amplitude = np.abs(fft_values) / n
        values[col] = {"frequency": freq, "amplitude": amplitude}
        ax = axes[i, 0]
        ax.plot(freq, amplitude)
        ax.set_title(f"Амплитудно-частотная характеристика: {col}")
        ax.set_xlabel("Частота")
        ax.set_ylabel("|X(f)| / N")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, values


def interpolation_results(df: pd.DataFrame, multiplier: int = 10) -> tuple[dict[str, Any], Any]:
    idx = np.arange(len(df), dtype=np.float64)
    x_new = np.linspace(idx[0], idx[-1], multiplier * len(idx))
    result: dict[str, Any] = {}
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 4 * len(df.columns)), squeeze=False)
    for i, col in enumerate(df.columns):
        y = df[col].to_numpy(dtype=np.float64)
        cubic = interp1d(idx, y, kind="cubic", fill_value="extrapolate")
        spline = UnivariateSpline(idx, y, k=3, s=0)
        y_cubic = cubic(x_new)
        y_spline = spline(x_new)
        result[col] = {"x_new": x_new, "cubic": y_cubic, "spline": y_spline}
        ax = axes[i, 0]
        ax.plot(idx, y, ".", markersize=2.5, label="исходные")
        ax.plot(x_new, y_cubic, label="кубическая", linewidth=1)
        ax.plot(x_new, y_spline, label="сплайн", linewidth=1, alpha=0.75)
        ax.set_title(f"Интерполяция: {col}")
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return result, fig


def binary_masks(df: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    masks: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        col_masks = {
            "positive": x > 0,
            "negative": x < 0,
            "zero": x == 0,
            "interval_-1_1": (x >= -1) & (x <= 1),
        }
        masks[col] = col_masks
        rows.append({"column": col, **{name: int(mask.sum()) for name, mask in col_masks.items()}})
    return masks, pd.DataFrame(rows).set_index("column")
