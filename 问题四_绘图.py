"""问题四移动边界结果绘图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook
from matplotlib.ticker import ScalarFormatter

import 问题4_求解 as problem4


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result4.xlsx"
DIAGNOSTICS_FILE = PROJECT_ROOT / "附件" / "附件3" / "result4_diagnostics.json"
INTERNAL_FIELD_FILE = PROJECT_ROOT / "附件" / "附件3" / "result4_internal_field.npz"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题四"
CRITICAL_MOISTURE = 0.15

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def read_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workbook = load_workbook(RESULT_FILE, data_only=True, read_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    fixed_radius_cm = np.asarray(rows[0][1:-1], dtype=float)
    times_h = np.asarray([row[0] for row in rows[1:]], dtype=float) / 3600.0
    moisture = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    return times_h, fixed_radius_cm, moisture


def read_internal_field(
    path: Path = INTERNAL_FIELD_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取求解器直接保存的参考域细网格结果。"""
    with np.load(path) as data:
        times_h = np.asarray(data["times_s"], dtype=float) / 3600.0
        xi_centers = np.asarray(data["xi_centers"], dtype=float)
        moisture = np.asarray(data["moisture"], dtype=float)
        surface_radii_cm = (
            100.0 * np.asarray(data["surface_radii_m"], dtype=float)
        )
    if moisture.shape != (len(times_h), len(xi_centers)):
        raise ValueError("内部细网格含水率数组尺寸与时间、空间坐标不一致。")
    if len(surface_radii_cm) != len(times_h):
        raise ValueError("内部细网格半径序列与时间序列长度不一致。")
    return times_h, xi_centers, moisture, surface_radii_cm


def threshold_times_from_output(
    times_h: np.ndarray,
    maximum_moisture: np.ndarray,
    threshold: float = CRITICAL_MOISTURE,
) -> tuple[float, float]:
    indices = np.flatnonzero(maximum_moisture < threshold)
    if len(indices) == 0:
        raise ValueError("输出记录内没有严格低于阈值的时刻。")
    index = int(indices[0])
    if index == 0:
        raise ValueError("首条输出已经低于阈值，无法夹逼交点。")
    previous_value = float(maximum_moisture[index - 1])
    current_value = float(maximum_moisture[index])
    crossing = float(times_h[index - 1]) + (
        (previous_value - threshold) / (previous_value - current_value)
    ) * float(times_h[index] - times_h[index - 1])
    return crossing, float(times_h[index])


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def verified_mechanism_cases(diagnostics: dict) -> list[dict]:
    """仅从明确完成整套复算的诊断中读取四组合对照。"""
    if not diagnostics.get("verification_complete", False):
        return []
    cases_by_name = {
        case["name"]: case
        for case in [
            diagnostics["baseline"],
            *diagnostics["mechanism_comparison"],
        ]
    }
    ordered_names = [
        "appendix3_fixed_radius",
        "appendix3_moving_radius",
        "appendix4_fixed_radius",
        "appendix4_moving_radius",
    ]
    return [cases_by_name[name] for name in ordered_names]


def main() -> None:
    times, _, moisture = read_result()
    internal_times, xi_centers, internal_moisture, surface_radius_cm = (
        read_internal_field()
    )
    if not np.allclose(internal_times, times, rtol=0.0, atol=1.0e-12):
        raise ValueError("内部细网格数据与 result4.xlsx 的输出时刻不一致。")
    history = problem4.read_radius_history(problem4.RADIUS_FILE)
    center_moisture = moisture[:, 0]
    surface_moisture = moisture[:, -1]
    maximum_moisture = np.maximum(
        moisture.max(axis=1), internal_moisture.max(axis=1)
    )
    continuous_time, discrete_time = threshold_times_from_output(
        times, maximum_moisture
    )
    threshold_index = int(np.flatnonzero(maximum_moisture < CRITICAL_MOISTURE)[0])

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(history.times / 3600.0, history.radii * 100.0, "o-", ms=2.5)
    axis.axhline(history.radii[-1] * 100.0, color="#d62728", ls="--",
                 label=f"72 h 末值 {history.radii[-1] * 100.0:.3f} cm")
    axis.set(xlabel="时间 / h", ylabel="药材半径 / cm", title="附件二药材半径变化")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "药材半径收缩曲线")

    selected_hours = np.array([6.0, 12.0, 24.0, 36.0, 48.0, discrete_time])
    selected_hours = selected_hours[selected_hours <= times[-1] + 1.0e-9]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for selected in selected_hours:
        index = int(np.argmin(np.abs(times - selected)))
        radii = np.concatenate(
            [
                [0.0],
                xi_centers * surface_radius_cm[index],
                [surface_radius_cm[index]],
            ]
        )
        profile = np.concatenate(
            [
                [center_moisture[index]],
                internal_moisture[index],
                [surface_moisture[index]],
            ]
        )
        label = (
            f"{times[index]:.4f} h（首次离散达标）"
            if index == threshold_index
            else f"{times[index]:.0f} h"
        )
        axis.plot(radii, profile, lw=1.4, label=label)
    axis.axhline(CRITICAL_MOISTURE, color="black", ls="--", lw=1.0,
                 label="阈值 0.15")
    axis.set(xlabel="到药材中心的实际距离 / cm",
             ylabel="含水率 / kg·kg$^{-1}$", title="收缩过程含水率径向剖面")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    save_figure(figure, "收缩过程含水率径向剖面")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(times, center_moisture, color="#d62728", label="中心")
    axis.plot(times, surface_moisture, color="#1f77b4", label="移动表面")
    axis.axhline(CRITICAL_MOISTURE, color="black", ls="--", label="阈值 0.15")
    axis.axvline(continuous_time, color="#ff7f0e", ls="-.",
                 label=f"t*={continuous_time:.4f} h")
    axis.axvline(discrete_time, color="#555555", ls=":",
                 label=f"t60={discrete_time:.4f} h")
    axis.set(xlabel="时间 / h", ylabel="含水率 / kg·kg$^{-1}$",
             title="中心与移动表面含水率演化")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "中心与移动表面含水率")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(times, maximum_moisture, color="#2c3e50")
    axis.axhline(CRITICAL_MOISTURE, color="#d62728", ls="--", label="阈值 0.15")
    axis.axvline(continuous_time, color="#ff7f0e", ls="-.",
                 label=f"t*={continuous_time:.4f} h")
    axis.axvline(discrete_time, color="#555555", ls=":",
                 label=f"t60={discrete_time:.4f} h")
    axis.set(xlabel="时间 / h", ylabel="全域最大含水率 / kg·kg$^{-1}$",
             title="移动域全域最大含水率")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "移动域全域最大含水率")

    left = max(threshold_index - 20, 0)
    right = min(threshold_index + 20, len(times) - 1)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(times[left:right + 1], maximum_moisture[left:right + 1],
              "o-", ms=3, color="#2c3e50")
    axis.axhline(CRITICAL_MOISTURE, color="#d62728", ls="--")
    axis.axvline(continuous_time, color="#ff7f0e", ls="-.",
                 label=f"t*={continuous_time:.4f} h")
    axis.axvline(discrete_time, color="#555555", ls=":",
                 label=f"t60={discrete_time:.4f} h")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.yaxis.set_major_formatter(formatter)
    axis.set(xlabel="时间 / h", ylabel="全域最大含水率 / kg·kg$^{-1}$",
             title="第四问达标时间局部判定")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "达标时间局部判定")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    image = axis.imshow(
        internal_moisture.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0.0, 1.0],
        cmap="YlOrRd",
        vmin=CRITICAL_MOISTURE,
        vmax=float(internal_moisture.max()),
    )
    figure.colorbar(image, ax=axis, label="含水率 / kg·kg$^{-1}$")
    axis.axvline(discrete_time, color="black", ls="--", lw=1.0)
    axis.set(xlabel="时间 / h", ylabel="归一化半径 ξ=r/R(t)",
             title="移动参考域含水率时空分布")
    save_figure(figure, "移动参考域含水率时空分布")

    if DIAGNOSTICS_FILE.exists():
        with DIAGNOSTICS_FILE.open("r", encoding="utf-8") as stream:
            diagnostics = json.load(stream)
        cases = verified_mechanism_cases(diagnostics)
    else:
        cases = []
    if cases:
        labels = [case["label"] for case in cases]
        values = [case["continuous_threshold_time_h"] for case in cases]
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        bars = axis.bar(
            labels,
            values,
            color=["#7f8c8d", "#95a5a6", "#1f77b4", "#d62728"],
        )
        axis.set(ylabel="连续达标时间 t* / h", title="物性变化与径向收缩的机制对照")
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2.0, value,
                      f"{value:.3f} h", ha="center", va="bottom")
        save_figure(figure, "物性与收缩机制对照")

    print(f"已生成 {len(list(FIGURE_DIR.glob('*.png')))} 张 PNG 图：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
