from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def check_data_quality(
    df: pd.DataFrame,
    expected_numeric: list[str] | None = None,
    expected_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Проверяет NaN, некорректные значения и несовпадения ожидаемых типов."""
    expected_numeric = expected_numeric or []
    report: dict[str, Any] = {
        "shape": list(df.shape),
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "invalid_numeric": {},
        "type_mismatches": {},
    }

    for col in expected_numeric:
        if col not in df.columns:
            report["invalid_numeric"][col] = ["СТОЛБЕЦ ОТСУТСТВУЕТ"]
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        bad_mask = converted.isna() & df[col].notna()
        if bad_mask.any():
            examples = df.loc[bad_mask, col].astype(str).head(10).tolist()
            report["invalid_numeric"][col] = examples
        if not pd.api.types.is_numeric_dtype(df[col]):
            report["type_mismatches"][col] = {
                "actual": str(df[col].dtype),
                "expected": "numeric",
            }

    if expected_types:
        for col, expected in expected_types.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if expected not in actual:
                report["type_mismatches"][col] = {"actual": actual, "expected": expected}

    return report


def print_quality_warnings(report: dict[str, Any]) -> None:
    missing = {k: v for k, v in report["missing"].items() if v}
    invalid = report["invalid_numeric"]
    mismatches = report["type_mismatches"]

    if not missing and not invalid and not mismatches:
        print("Проблемы качества данных не обнаружены.")
        return

    print("\n[ПРЕДУПРЕЖДЕНИЕ] Обнаружены проблемы качества данных:")
    if missing:
        print("  Пустые ячейки:", missing)
    if invalid:
        print("  Некорректные числовые значения:", invalid)
    if mismatches:
        print("  Несовпадения типов:", mismatches)


def show_table(df: pd.DataFrame, full: bool = True) -> None:
    """Выводит содержимое таблицы, форму и типы."""
    with pd.option_context("display.max_rows", None if full else 20, "display.max_columns", None, "display.width", 160):
        print(df.to_string(index=False) if full else df.head(20).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print("Dtypes:")
    print(df.dtypes)


def extract_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df.loc[:, columns].copy()


def cast_types(df: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
    result = df.copy()
    for col, dtype in type_map.items():
        result[col] = result[col].astype(dtype)
    return result
