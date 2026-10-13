from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from src.config import OUTPUT_DIR, timestamped_path


def _sniff_delimiter_and_header(path: Path, default_sep: str = ",") -> tuple[str, bool]:
    """Пытается определить разделитель и наличие заголовка для текстового файла."""
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t |")
        sep = dialect.delimiter
    except csv.Error:
        sep = default_sep
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = True
    return sep, has_header


def _auto_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        df = df.copy()
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def read_table(path: str | Path, header: str | int | None = "auto") -> pd.DataFrame:
    """Читает .xlsx/.csv/.txt/.mat и автоматически именует столбцы без заголовков."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()

    if ext == ".csv":
        sep, detected_header = _sniff_delimiter_and_header(p, ",")
        hdr = 0 if (header == "auto" and detected_header) else None if header == "auto" else header
        df = pd.read_csv(p, sep=sep, header=hdr)
    elif ext == ".txt":
        sep, detected_header = _sniff_delimiter_and_header(p, "\t")
        hdr = 0 if (header == "auto" and detected_header) else None if header == "auto" else header
        # Если Sniffer не справился, pandas сможет разобрать несколько пробелов.
        if sep == " ":
            df = pd.read_csv(p, sep=r"\s+", engine="python", header=hdr)
        else:
            df = pd.read_csv(p, sep=sep, header=hdr)
    elif ext == ".xlsx":
        # Для Excel надежное автоматическое определение заголовка неоднозначно;
        # header="auto" трактуем как обычную первую строку-заголовок.
        hdr = 0 if header == "auto" else header
        df = pd.read_excel(p, header=hdr)
    elif ext == ".mat":
        mat = loadmat(p)
        if "data" in mat:
            arr = np.asarray(mat["data"])
            columns = None
            if "columns" in mat:
                raw_cols = np.asarray(mat["columns"]).ravel()
                columns = []
                for c in raw_cols:
                    if isinstance(c, np.ndarray):
                        c = c.item() if c.size == 1 else "".join(map(str, c.ravel()))
                    columns.append(str(c).strip())
                if len(columns) != arr.shape[1]:
                    columns = None
            df = pd.DataFrame(arr, columns=columns)
        else:
            arrays = {
                k: v for k, v in mat.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2
            }
            if not arrays:
                raise ValueError("В MAT-файле не найден двумерный массив")
            key = next(iter(arrays))
            df = pd.DataFrame(arrays[key])
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    return _auto_name_columns(df)


def write_table(df: pd.DataFrame, base: str, ext: str, output_dir: Path | None = None) -> Path:
    """Записывает DataFrame в .xlsx/.csv/.txt/.mat с сохранением заголовков."""
    ext = ext.lower().lstrip(".")
    path = timestamped_path(base, ext, output_dir or OUTPUT_DIR)

    if ext == "xlsx":
        df.to_excel(path, index=False)
    elif ext == "csv":
        df.to_csv(path, index=False)
    elif ext == "txt":
        df.to_csv(path, sep="\t", index=False)
    elif ext == "mat":
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
            raise TypeError("Для записи в MAT в этой лабораторной используются только числовые столбцы")
        savemat(path, {
            "data": df.to_numpy(dtype=np.float64),
            "columns": np.asarray(df.columns.astype(str), dtype=object),
        })
    else:
        raise ValueError(f"Неподдерживаемый формат: .{ext}")
    return path


def write_all_formats(df: pd.DataFrame, base: str = "table") -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for ext in ("xlsx", "csv", "txt", "mat"):
        paths[ext] = write_table(df, base, ext)
    return paths


def verify_roundtrip(paths: Iterable[Path], expected_columns: list[str]) -> dict[str, tuple[int, int]]:
    """Повторно читает сохраненные файлы и возвращает их формы."""
    result: dict[str, tuple[int, int]] = {}
    for p in paths:
        df = read_table(p)
        if list(df.columns) != expected_columns:
            raise AssertionError(f"После чтения {p.name} заголовки отличаются: {list(df.columns)}")
        result[p.suffix.lower()] = df.shape
    return result
