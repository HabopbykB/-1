from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, rotate
from sklearn.decomposition import PCA

from src.config import SEED, save_figure, timestamped_path


def load_rgb_image(path: str | Path) -> np.ndarray:
    img = mpimg.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def split_channels(img: np.ndarray) -> dict[str, np.ndarray]:
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается цветное изображение HxWx3")
    return {"R": img[..., 0].copy(), "G": img[..., 1].copy(), "B": img[..., 2].copy()}


def rotate_image(img: np.ndarray, angle: float = 40.0) -> np.ndarray:
    out = rotate(img.astype(np.float32), angle=angle, reshape=True, order=3, mode="constant", cval=0)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def gray_histogram(gray: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    return np.histogram(gray.ravel(), bins=bins, range=(0, 256))


def split_by_histogram(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    # Квантили вычисляются по распределению яркости (эквивалентно накопленной гистограмме).
    t1, t2 = np.quantile(gray, [1 / 3, 2 / 3]).astype(int)
    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)
    return part1, part2, part3, (int(t1), int(t2))


def add_circular_frame(img: np.ndarray, thickness: int = 5) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0 - 2
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outside = distance > radius
    border = (distance >= radius - thickness) & (distance <= radius)
    framed = img.copy()
    framed[outside] = 0
    framed[border] = 0
    return framed


def add_noise(img: np.ndarray, sigma: float = 25.0, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    # sigma=0 по каналу цвета предотвращает смешивание RGB-каналов.
    blurred = gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0))
    return np.clip(blurred, 0, 255).astype(np.uint8)


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    for channel in range(3):
        out[..., channel] = convolve(img[..., channel].astype(np.float32), kernel, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def pca_image(img: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    pixels = img.reshape(-1, 3).astype(np.float64)
    n_components = min(n_components, pixels.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    reduced = pca.fit_transform(pixels)
    restored = pca.inverse_transform(reduced).reshape(img.shape)
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    return restored, pca.explained_variance_ratio_


def _comparison_figure(title: str, images: list[tuple[str, np.ndarray]], cmap: str | None = None):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, arr) in zip(axes, images):
        ax.imshow(arr, cmap=cmap if arr.ndim == 2 else None)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def run_image_lab(path: str | Path) -> dict[str, Any]:
    img = load_rgb_image(path)
    channels = split_channels(img)
    for name, arr in channels.items():
        np.save(timestamped_path(f"task50_channel_{name}", "npy"), arr)

    fig = plt.figure(figsize=(16, 4))
    axes = fig.subplots(1, 4)
    axes[0].imshow(img); axes[0].set_title("Исходное")
    cmaps = {"R": "Reds", "G": "Greens", "B": "Blues"}
    for ax, key in zip(axes[1:], ["R", "G", "B"]):
        ax.imshow(channels[key], cmap=cmaps[key]); ax.set_title(f"Канал {key}")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); channels_path = save_figure(fig, "task50_channels"); plt.close(fig)

    rotated = rotate_image(img, 40)
    fig = _comparison_figure("Поворот на 40°", [("Исходное", img), ("Поворот", rotated)])
    rotated_path = save_figure(fig, "task50_rotated"); plt.close(fig)

    gray = to_grayscale(img)
    fig = _comparison_figure("Преобразование в ч/б", [("Исходное", img), ("Ч/б", gray)], cmap="gray")
    gray_path = save_figure(fig, "task50_gray"); plt.close(fig)

    hist, edges = gray_histogram(gray)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(edges[:-1], hist, width=1.0, align="edge")
    ax.set_xlim(0, 255); ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость"); ax.set_ylabel("Частота")
    hist_path = save_figure(fig, "task50_gray_hist"); plt.close(fig)

    p1, p2, p3, thresholds = split_by_histogram(gray)
    fig = _comparison_figure(
        f"Разбиение по гистограмме, пороги {thresholds}",
        [("Темные", p1), ("Средние", p2), ("Светлые", p3)],
        cmap="gray",
    )
    split_path = save_figure(fig, "task50_split3"); plt.close(fig)

    framed = add_circular_frame(img)
    fig = _comparison_figure("Черная круговая рамка", [("Исходное", img), ("Рамка", framed)])
    frame_path = save_figure(fig, "task50_circular_frame"); plt.close(fig)

    noisy = add_noise(img)
    fig = _comparison_figure("Случайный шум", [("Исходное", img), ("Шум", noisy)])
    noise_path = save_figure(fig, "task50_noise"); plt.close(fig)

    blurred = gaussian_blur(img)
    fig = _comparison_figure("Гауссовское размытие", [("Исходное", img), ("Размытие", blurred)])
    blur_path = save_figure(fig, "task50_blur"); plt.close(fig)

    sharp = sharpen(img)
    fig = _comparison_figure("Sharpening", [("Исходное", img), ("Sharpen", sharp)])
    sharp_path = save_figure(fig, "task50_sharpen"); plt.close(fig)

    pca_img, explained = pca_image(img, n_components=2)
    fig = _comparison_figure(
        f"PCA изображения, сумма объясненной дисперсии={explained.sum():.4f}",
        [("Исходное", img), ("PCA → inverse", pca_img)],
    )
    pca_path = save_figure(fig, "task50_pca"); plt.close(fig)

    return {
        "shape": list(img.shape),
        "histogram_thresholds": thresholds,
        "pca_explained_variance_ratio": explained,
        "saved_figures": [
            channels_path, rotated_path, gray_path, hist_path, split_path,
            frame_path, noise_path, blur_path, sharp_path, pca_path,
        ],
    }
