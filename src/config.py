from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SEED = 42
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
DATASET_PATH = DATA_DIR / "dataset_var2.csv"
IMAGE_PATH = DATA_DIR / "sample_image.png"

# По примеру из методички для варианта оставляем только исходные числовые
# столбцы x1, x2 и y. Служебные id/timestamp/category и x3 не изменяют
# исходный CSV и не входят в основной массив X.
VARIANT_COLUMNS = ["x1", "x2", "y"]
EXPECTED_NUMERIC_COLUMNS = ["x1", "x2", "x3", "y"]
ALPHA = 0.05

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Фиксирует seed для Python, NumPy, TensorFlow и PyTorch (если доступны)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def timestamped_path(base: str, ext: str, output_dir: Path | None = None) -> Path:
    """Возвращает путь с датой и временем в имени файла."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ext.lstrip(".")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return out_dir / f"{base}_{ts}.{ext}"


def save_figure(fig, base: str, dpi: int = 150) -> Path:
    """Сохраняет matplotlib Figure в PNG с датой и временем."""
    path = timestamped_path(base, "png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(data: Any, base: str) -> Path:
    path = timestamped_path(base, "json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def save_text(text: str, base: str) -> Path:
    path = timestamped_path(base, "txt")
    path.write_text(text, encoding="utf-8")
    return path


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
