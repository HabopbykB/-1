from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import check_data_quality, print_quality_warnings


def handle_problem(
    df: pd.DataFrame,
    col: str,
    action: str,
    value: Any | None = None,
) -> pd.DataFrame:
    """Обрабатывает проблему в столбце средствами Pandas."""
    if col not in df.columns:
        raise KeyError(col)
    result = df.copy()
    action = action.strip().lower()

    if action == "drop":
        result = result.dropna(subset=[col])
    elif action == "cast":
        result[col] = pd.to_numeric(result[col], errors="coerce")
    elif action == "mean":
        numeric = pd.to_numeric(result[col], errors="coerce")
        result[col] = numeric.fillna(numeric.mean())
    elif action == "ffill":
        result[col] = result[col].ffill()
    elif action == "bfill":
        result[col] = result[col].bfill()
    elif action == "zero":
        result[col] = result[col].fillna(0)
    elif action == "value":
        result[col] = result[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")
    return result


def interactive_menu(df: pd.DataFrame, expected_numeric: list[str]) -> pd.DataFrame:
    """Интерактивное меню из задания 8."""
    result = df.copy()
    while True:
        print("\n=== Меню обработки проблем ===")
        print("1. Показать проблемы")
        print("2. Обработать столбец")
        print("3. Завершить обработку")
        choice = input("Выбор: ").strip()

        if choice == "1":
            report = check_data_quality(result, expected_numeric=expected_numeric)
            print_quality_warnings(report)
        elif choice == "2":
            col = input(f"Столбец {list(result.columns)}: ").strip()
            action = input("Действие (drop/cast/mean/ffill/bfill/zero/value): ").strip().lower()
            value = None
            if action == "value":
                raw = input("Значение: ").strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            result = handle_problem(result, col, action, value)
            print("Готово.")
        elif choice == "3":
            return result
        else:
            print("Неизвестный пункт меню.")


def auto_clean_numeric(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Автоматический сценарий для полного запуска: cast -> заполнение средним -> float64."""
    result = df.copy()
    log: list[str] = []
    for col in numeric_columns:
        before = result[col].copy()
        converted = pd.to_numeric(result[col], errors="coerce")
        invalid_count = int((converted.isna() & before.notna()).sum())
        if invalid_count:
            log.append(f"{col}: приведено к числу; некорректных значений -> NaN: {invalid_count}")
        missing_count = int(converted.isna().sum())
        if missing_count:
            mean_value = float(converted.mean())
            converted = converted.fillna(mean_value)
            log.append(f"{col}: {missing_count} пустых/некорректных значений заполнено средним {mean_value:.6f}")
        result[col] = converted.astype("float64")
    return result, log


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float64)


@dataclass
class MinMaxScalerCustom:
    a: float = 0.0
    b: float = 1.0
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MinMaxScalerCustom":
        arr = np.asarray(x, dtype=np.float64)
        self.min_ = np.min(arr, axis=0)
        self.max_ = np.max(arr, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        arr = np.asarray(x, dtype=np.float64)
        spread = self.max_ - self.min_
        safe = np.where(spread == 0, 1.0, spread)
        scaled = self.a + (arr - self.min_) * (self.b - self.a) / safe
        if np.ndim(spread) == 0:
            return np.asarray(self.a if spread == 0 else scaled)
        constant_cols = spread == 0
        if np.any(constant_cols):
            scaled[..., constant_cols] = self.a
        return scaled

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        if self.b == self.a:
            raise ValueError("Границы a и b не должны совпадать")
        arr = np.asarray(x_new, dtype=np.float64)
        spread = self.max_ - self.min_
        restored = (arr - self.a) * spread / (self.b - self.a) + self.min_
        return restored

    def parameters(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "min": self.min_, "max": self.max_}


@dataclass
class StandardScalerCustom:
    ddof: int = 0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerCustom":
        arr = np.asarray(x, dtype=np.float64)
        self.mean_ = np.mean(arr, axis=0)
        self.std_ = np.std(arr, axis=0, ddof=self.ddof)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        arr = np.asarray(x, dtype=np.float64)
        safe = np.where(self.std_ == 0, 1.0, self.std_)
        return (arr - self.mean_) / safe

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        arr = np.asarray(x_new, dtype=np.float64)
        safe = np.where(self.std_ == 0, 1.0, self.std_)
        return arr * safe + self.mean_

    def parameters(self) -> dict[str, Any]:
        return {"ddof": self.ddof, "mean": self.mean_, "std": self.std_}


def split_train_val_test(
    x: np.ndarray,
    ratios: tuple[float, float, float] = (70, 15, 15),
    percent: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(ratios, dtype=np.float64)
    if np.any(r < 0) or np.isclose(r.sum(), 0):
        raise ValueError("Доли должны быть неотрицательными и иметь ненулевую сумму")
    if percent:
        r = r / 100.0
    r = r / r.sum()

    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    return x_train, x_val, x_test
