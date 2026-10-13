from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import (
    ALPHA,
    DATASET_PATH,
    EXPECTED_NUMERIC_COLUMNS,
    IMAGE_PATH,
    OUTPUT_DIR,
    SEED,
    VARIANT_COLUMNS,
    save_figure,
    save_json,
    save_text,
    set_seeds,
    timestamped_path,
)
from src.data_loader import cast_types, check_data_quality, extract_columns, print_quality_warnings, show_table
from src.image_proc import run_image_lab
from src.io_utils import read_table, verify_roundtrip, write_all_formats, write_table
from src.matrix_ops import build_feature_tensor, create_sparse_array, inverse_random_matrix, pca_pipeline_search
from src.preprocess import (
    MinMaxScalerCustom,
    StandardScalerCustom,
    auto_clean_numeric,
    interactive_menu,
    split_train_val_test,
    to_numpy,
)
from src.scaling import compare_scalers
from src.stats_analysis import (
    binary_masks,
    confidence_intervals,
    correlation_significance,
    covariance_correlation,
    derivatives_and_gradients,
    distribution_hypotheses,
    interpolation_results,
    norms,
    pairwise_convolutions,
    plot_cross_correlations,
    plot_derivatives,
    plot_ecdfs,
    plot_fft_amplitudes,
    plot_histograms,
    plot_periodograms,
    plot_series,
    plot_spectrograms,
    plot_statistics,
    scalar_and_vector_products,
    sorted_columns,
    statistics_table,
)
from src.tensors import (
    create_tpq,
    keras_operation,
    make_pytorch_tensor,
    make_tensorflow_tensor,
    pytorch_axwb,
    pytorch_operation,
    tensorflow_axwb,
)
from src.windows import moving_average, sliding_window


def section(number: str, title: str) -> None:
    print("\n" + "=" * 90)
    print(f"ЗАДАНИЕ {number}. {title}")
    print("=" * 90)


def save_df(df: pd.DataFrame, base: str) -> Path:
    return write_table(df, base, "csv")


def save_npz(base: str, **arrays) -> Path:
    path = timestamped_path(base, "npz")
    np.savez_compressed(path, **arrays)
    return path


def tensor_shape(obj) -> list[int]:
    shape = getattr(obj, "shape", ())
    return [int(v) for v in shape]


def run_lab(interactive_cleaning: bool = False, brief: bool = False) -> None:
    set_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[str] = []
    summary.append(f"Лабораторная работа 1, dataset_var2, seed={SEED}")
    summary.append(f"Столбцы варианта: {VARIANT_COLUMNS}")

    # ------------------------------------------------------------------
    section("1", "Считывание файлов .xlsx/.csv/.txt/.mat")
    df_raw = read_table(DATASET_PATH)
    print(f"Исходный файл: {DATASET_PATH}")
    print(f"Форма: {df_raw.shape}; столбцы: {list(df_raw.columns)}")
    summary.append(f"Исходный датасет: shape={df_raw.shape}")

    # ------------------------------------------------------------------
    section("2-3", "Запись таблиц и сохранение изображений")
    print("Функции write_table()/write_all_formats() и save_figure() готовы.")
    print("Таблицы будут сохранены после очистки числового варианта во всех 4 форматах,")
    print("графики заданий 14-50 автоматически сохраняются в PNG с датой/временем.")

    # ------------------------------------------------------------------
    section("4", "Проверка пустых ячеек, некорректных данных и типов")
    quality_before = check_data_quality(df_raw, expected_numeric=EXPECTED_NUMERIC_COLUMNS)
    print_quality_warnings(quality_before)
    save_json(quality_before, "task04_quality_before")
    summary.append(f"До очистки: missing x2={quality_before['missing'].get('x2', 0)}")
    summary.append(f"До очистки: invalid x1={len(quality_before['invalid_numeric'].get('x1', []))}")

    # ------------------------------------------------------------------
    section("5", "Вывод таблицы")
    show_table(df_raw, full=not brief)

    # ------------------------------------------------------------------
    section("6", "Извлечение столбцов по варианту")
    df_variant = extract_columns(df_raw, VARIANT_COLUMNS)
    print(df_variant.head(10).to_string(index=False))
    print("Исходный df не изменяется; x3 и служебные столбцы в X не входят.")

    # ------------------------------------------------------------------
    section("7", "Конкретные типы Pandas")
    print("Типы до обработки:")
    print(df_variant.dtypes)
    print("Ожидаемые после обработки:", {c: "float64" for c in VARIANT_COLUMNS})

    # ------------------------------------------------------------------
    section("8", "Предобработка и меню действий")
    if interactive_cleaning:
        df_clean = interactive_menu(df_variant, VARIANT_COLUMNS)
        # Чтобы дальнейшие задания гарантированно получили числовой массив,
        # после выхода из меню оставшиеся проблемы добираются безопасным авто-сценарием.
        df_clean, auto_log = auto_clean_numeric(df_clean, VARIANT_COLUMNS)
    else:
        df_clean, auto_log = auto_clean_numeric(df_variant, VARIANT_COLUMNS)
    for line in auto_log:
        print("-", line)
    df_clean = cast_types(df_clean, {c: "float64" for c in VARIANT_COLUMNS})
    quality_after = check_data_quality(df_clean, expected_numeric=VARIANT_COLUMNS)
    print_quality_warnings(quality_after)
    save_json({"actions": auto_log, "after": quality_after}, "task08_cleaning")

    # Теперь демонстрируем реальную запись/чтение всех форматов из заданий 1-2.
    format_paths = write_all_formats(df_clean, "task02_variant_clean")
    roundtrip = verify_roundtrip(format_paths.values(), list(df_clean.columns))
    print("Сохранены форматы:", {k: str(v.name) for k, v in format_paths.items()})
    print("Контрольное чтение:", roundtrip)
    summary.append(f"После очистки: shape={df_clean.shape}, NaN={int(df_clean.isna().sum().sum())}")

    # ------------------------------------------------------------------
    section("9", "DataFrame -> NumPy")
    x = to_numpy(df_clean)
    print("X shape:", x.shape, "dtype:", x.dtype)

    # ------------------------------------------------------------------
    section("10-11", "Собственное масштабирование и обратное преобразование")
    mm01 = MinMaxScalerCustom(0, 1).fit(x)
    mmab = MinMaxScalerCustom(-1, 1).fit(x)
    zsc = StandardScalerCustom(ddof=0).fit(x)
    x_mm01 = mm01.transform(x)
    x_mmab = mmab.transform(x)
    x_z = zsc.transform(x)
    scale_params = {
        "minmax_0_1": mm01.parameters(),
        "minmax_-1_1": mmab.parameters(),
        "zscore": zsc.parameters(),
    }
    params_path = save_json(scale_params, "task10_scaling_parameters")
    inv_errors = {
        "minmax_0_1": float(np.max(np.abs(mm01.inverse_transform(x_mm01) - x))),
        "minmax_-1_1": float(np.max(np.abs(mmab.inverse_transform(x_mmab) - x))),
        "zscore": float(np.max(np.abs(zsc.inverse_transform(x_z) - x))),
    }
    print("Параметры сохранены:", params_path.name)
    print("Максимальные ошибки восстановления:", inv_errors)

    # ------------------------------------------------------------------
    section("12", "Разбиение train/val/test")
    x_train, x_val, x_test = split_train_val_test(x, (70, 15, 15), percent=True)
    print("train/val/test:", x_train.shape, x_val.shape, x_test.shape)
    summary.append(f"split: {len(x_train)}/{len(x_val)}/{len(x_test)}")

    # ------------------------------------------------------------------
    section("13", "Фиксация seed")
    print(f"SEED={SEED} зафиксирован для random/NumPy/TensorFlow/PyTorch (если доступны).")

    # ------------------------------------------------------------------
    section("14", "График исходных данных")
    fig = plot_series(x, VARIANT_COLUMNS); print(save_figure(fig, "task14_original_data")); plt.close(fig)

    section("15", "Нормализованная гистограмма")
    fig = plot_histograms(x, VARIANT_COLUMNS); print(save_figure(fig, "task15_histograms")); plt.close(fig)

    section("16", "Сортировка столбцов")
    sorted_data = sorted_columns(df_clean)
    for col, arr in sorted_data.items():
        print(f"\n{col} sorted:")
        print(arr if not brief else arr[:30])
    sorted_df = pd.DataFrame(sorted_data)
    save_df(sorted_df, "task16_sorted_columns")

    section("17", "Эмпирическая функция распределения")
    fig = plot_ecdfs(x, VARIANT_COLUMNS); print(save_figure(fig, "task17_ecdf")); plt.close(fig)

    section("18", "Среднее, дисперсия, мода, медиана")
    stats_df = statistics_table(df_clean)
    print(stats_df)
    save_df(stats_df.reset_index(), "task18_statistics")
    fig = plot_statistics(stats_df); print(save_figure(fig, "task18_statistics_plot")); plt.close(fig)

    section("19", "95% доверительные интервалы")
    ci_df = confidence_intervals(df_clean, ALPHA)
    print(ci_df)
    save_df(ci_df.reset_index(), "task19_confidence_intervals")

    section("20", "Ковариация и корреляция")
    covcorr = covariance_correlation(df_clean)
    print("NumPy covariance:\n", covcorr["numpy_cov"])
    print("NumPy correlation:\n", covcorr["numpy_corr"])
    save_df(covcorr["pandas_cov"].reset_index(), "task20_covariance")
    save_df(covcorr["pandas_corr"].reset_index(), "task20_correlation")

    section("21", "Значимость коэффициентов корреляции")
    sig_df = correlation_significance(df_clean, ALPHA)
    print(sig_df.to_string(index=False))
    save_df(sig_df, "task21_correlation_significance")

    section("22", "Взаимная корреляция от числа сдвигов")
    fig, xcorr = plot_cross_correlations(df_clean, max_lags=50)
    print(save_figure(fig, "task22_cross_correlation")); plt.close(fig)
    npz_payload = {}
    for pair, values in xcorr.items():
        npz_payload[f"{pair}_lags"] = values["lags"]
        npz_payload[f"{pair}_corr"] = values["correlation"]
    save_npz("task22_cross_correlation_values", **npz_payload)

    section("23", "Численная производная и градиент")
    deriv_df, grad_df = derivatives_and_gradients(df_clean)
    save_df(deriv_df, "task23_derivative")
    save_df(grad_df, "task23_gradient")
    fig = plot_derivatives(deriv_df, grad_df); print(save_figure(fig, "task23_derivative_gradient")); plt.close(fig)

    section("24", "Свёртка столбцов")
    convolutions = pairwise_convolutions(df_clean)
    print({k: v.shape for k, v in convolutions.items()})
    save_npz("task24_convolutions", **convolutions)

    section("25", "Скалярное и векторное произведения")
    products = scalar_and_vector_products(df_clean)
    print(products)
    save_json(products, "task25_products")

    section("26", "L1 и L2 нормы")
    norms_df = norms(df_clean)
    print(norms_df)
    save_df(norms_df.reset_index(), "task26_norms")

    section("27", "Гипотезы: равномерное и нормальное распределения")
    hypotheses = distribution_hypotheses(df_clean, ALPHA)
    print(json.dumps(hypotheses, ensure_ascii=False, indent=2))
    save_json(hypotheses, "task27_distribution_hypotheses")

    section("28", "Спектрограммы")
    fig = plot_spectrograms(df_clean); print(save_figure(fig, "task28_spectrograms")); plt.close(fig)

    section("29", "Периодограммы")
    fig, _ = plot_periodograms(df_clean); print(save_figure(fig, "task29_periodograms")); plt.close(fig)

    section("30", "АЧХ через FFT")
    fig, _ = plot_fft_amplitudes(df_clean); print(save_figure(fig, "task30_fft_amplitudes")); plt.close(fig)

    section("31-32", "Кубическая и сплайн-интерполяция")
    _, fig = interpolation_results(df_clean)
    print(save_figure(fig, "task31_32_interpolation")); plt.close(fig)

    section("33", "Бинарные маски")
    masks, mask_counts = binary_masks(df_clean)
    print(mask_counts)
    save_df(mask_counts.reset_index(), "task33_mask_counts")
    mask_arrays = {}
    for col, mdict in masks.items():
        for name, mask in mdict.items():
            mask_arrays[f"{col}__{name}"] = mask.astype(np.uint8)
    save_npz("task33_binary_masks", **mask_arrays)

    # ------------------------------------------------------------------
    section("34-35", "Сравнение масштабирования со sklearn и инверсия")
    scaling_df, fig, scaling_details = compare_scalers(x, VARIANT_COLUMNS)
    print(save_figure(fig, "task34_scaling_comparison")); plt.close(fig)
    save_df(scaling_df, "task34_scaling_values")
    print("Ошибки обратного преобразования:", scaling_details["inverse_errors"])
    save_json(scaling_details["inverse_errors"], "task35_inverse_errors")

    # ------------------------------------------------------------------
    section("36-37", "Скользящее окно и скользящее среднее")
    width = 5
    first_col = df_clean.columns[0]
    first_values = df_clean[first_col].to_numpy(dtype=np.float64)
    windows = sliding_window(first_values, width)
    ma = moving_average(first_values, width)
    print("Окна shape:", windows.shape, "moving average shape:", ma.shape)
    save_npz("task36_37_windows", windows=windows, moving_average=ma)

    # ------------------------------------------------------------------
    section("38-45", "TensorFlow, PyTorch и Keras")
    tensor_report: dict[str, object] = {}

    # PyTorch можно выполнить независимо от TensorFlow/Keras.
    try:
        x_pt = make_pytorch_tensor(x)
        pt_axwb = pytorch_axwb(x, seed=SEED)
        tensor_report["pytorch_X_shape"] = tensor_shape(x_pt)
        tensor_report["pytorch_AXWB_shape"] = tensor_shape(pt_axwb["result"])
        print("PyTorch X:", x_pt.shape, "AXW+B:", pt_axwb["result"].shape)
    except Exception as exc:
        tensor_report["pytorch_error"] = f"{type(exc).__name__}: {exc}"
        print("[ПРЕДУПРЕЖДЕНИЕ] PyTorch:", exc)

    try:
        x_tf = make_tensorflow_tensor(x)
        tf_axwb = tensorflow_axwb(x, seed=SEED)
        tensor_report["tensorflow_X_shape"] = tensor_shape(x_tf)
        tensor_report["tensorflow_AXWB_shape"] = tensor_shape(tf_axwb["result"])
        print("TensorFlow X:", x_tf.shape, "AXW+B:", tf_axwb["result"].shape)
    except Exception as exc:
        tensor_report["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
        print("[ПРЕДУПРЕЖДЕНИЕ] TensorFlow:", exc)

    t, p, q = create_tpq(SEED)
    save_npz("task43_T_P_Q", T=t, P=p, Q=q)
    try:
        v_k = keras_operation(t, p, q)
        v_k_np = v_k.numpy() if hasattr(v_k, "numpy") else np.asarray(v_k)
        tensor_report["keras_V_shape"] = list(v_k_np.shape)
        save_npz("task44_keras_V", V=v_k_np)
        print("Keras V:", v_k_np.shape)
    except Exception as exc:
        tensor_report["keras_error"] = f"{type(exc).__name__}: {exc}"
        print("[ПРЕДУПРЕЖДЕНИЕ] Keras:", exc)

    try:
        v_t = pytorch_operation(t, p, q)
        v_t_np = v_t.detach().cpu().numpy()
        tensor_report["pytorch_V_shape"] = list(v_t_np.shape)
        save_npz("task45_pytorch_V", V=v_t_np)
        print("PyTorch V:", v_t_np.shape)
    except Exception as exc:
        tensor_report["pytorch_V_error"] = f"{type(exc).__name__}: {exc}"
        print("[ПРЕДУПРЕЖДЕНИЕ] PyTorch operation:", exc)

    save_json(tensor_report, "task38_45_tensor_report")

    # ------------------------------------------------------------------
    section("46", "Разреженный массив SciPy 100000 x 100000")
    s = create_sparse_array(SEED, density=1e-5)
    sparse_path = timestamped_path("task46_sparse_100000x100000", "npz")
    sparse.save_npz(sparse_path, s)
    print("shape:", s.shape, "nnz:", s.nnz, "saved:", sparse_path.name)
    summary.append(f"sparse nnz={s.nnz}")

    section("47", "Обратная матрица 100 x 100")
    matrix, matrix_inv, inverse_error = inverse_random_matrix(SEED, 100)
    print("max |M*M_inv-I| =", inverse_error)
    save_npz("task47_matrix_inverse", matrix=matrix, inverse=matrix_inv)

    section("48", "Многомерный массив F")
    f_tensor, channel_names = build_feature_tensor(x)
    f_path = timestamped_path("task48_F_tensor", "npy")
    np.save(f_path, f_tensor)
    print("F shape:", f_tensor.shape)
    print("Каналы F:", channel_names)
    save_json({"shape": list(f_tensor.shape), "channels": channel_names}, "task48_F_description")

    section("49", "Pipeline sklearn: scaling + PCA")
    pca_result = pca_pipeline_search(f_tensor, SEED)
    comparison = pca_result["comparison"]
    print(comparison.to_string(index=False))
    print("Лучший вариант:", pca_result["best"])
    save_df(comparison, "task49_pca_comparison")
    print(save_figure(pca_result["fig_comparison"], "task49_pca_candidates")); plt.close(pca_result["fig_comparison"])
    print(save_figure(pca_result["fig_projection"], "task49_best_projection")); plt.close(pca_result["fig_projection"])
    print(save_figure(pca_result["fig_log_likelihood"], "task49_log_likelihood")); plt.close(pca_result["fig_log_likelihood"])
    ll_df = pd.DataFrame({
        "n_components": pca_result["log_likelihood_components"],
        "log_likelihood": pca_result["log_likelihood_values"],
    })
    save_df(ll_df, "task49_log_likelihood_values")
    summary.append(f"best PCA: {pca_result['best']}")

    # ------------------------------------------------------------------
    section("50", "Обработка цветного изображения средствами SciPy")
    image_result = run_image_lab(IMAGE_PATH)
    print("Изображение:", IMAGE_PATH.name, "shape:", image_result["shape"])
    print("PCA explained variance:", image_result["pca_explained_variance_ratio"])
    save_json(image_result, "task50_image_report")

    # ------------------------------------------------------------------
    section("ИТОГ", "Лабораторная работа выполнена")
    summary.append(f"output directory: {OUTPUT_DIR}")
    report_path = save_text("\n".join(summary), "lab1_run_summary")
    print("Все результаты сохранены в:", OUTPUT_DIR)
    print("Итоговый отчет:", report_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Лабораторная работа 1 — вариант dataset_var2")
    parser.add_argument(
        "--interactive-cleaning",
        action="store_true",
        help="Включить интерактивное меню обработки проблем из задания 8.",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Сократить вывод большой исходной/отсортированной таблицы (удобно для проверки).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_lab(interactive_cleaning=args.interactive_cleaning, brief=args.brief)
    except KeyboardInterrupt:
        print("\nВыполнение остановлено пользователем.")
        sys.exit(130)
