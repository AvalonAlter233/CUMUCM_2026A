"""问题三绘图脚本。

脚本只读取附件一和附件三中的 result3.xlsx，生成论文所需的 PNG 图。
所有路径均相对于本脚本所在目录，换电脑后无需修改绝对路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook
from matplotlib.ticker import ScalarFormatter


PROJECT_ROOT = Path(__file__).resolve().parent
BOUNDARY_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
RESULT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result3.xlsx"
DIAGNOSTICS_FILE = PROJECT_ROOT / "附件" / "附件3" / "result3_diagnostics.json"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题三"
CRITICAL_MOISTURE = 0.15

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def read_boundary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workbook = load_workbook(BOUNDARY_FILE, data_only=True, read_only=True)
    rows = [
        row for row in workbook.active.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    data = np.asarray(rows, dtype=float)
    return data[:, 0] / 3600.0, data[:, 1], data[:, 2]


def read_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workbook = load_workbook(RESULT_FILE, data_only=True, read_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    radius_cm = np.asarray(rows[0][1:], dtype=float)
    times_h = np.asarray([row[0] for row in rows[1:]], dtype=float) / 3600.0
    moisture = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    return times_h, radius_cm, moisture


def threshold_times_from_output(
    times_h: np.ndarray,
    maximum_moisture: np.ndarray,
    threshold: float = CRITICAL_MOISTURE,
) -> tuple[float, float]:
    """由 60 s 输出记录返回线性交点估计与首次严格达标记录。"""
    indices = np.flatnonzero(maximum_moisture < threshold)
    if len(indices) == 0:
        raise ValueError("输出记录内没有严格低于阈值的时刻。")
    index = int(indices[0])
    if index == 0:
        raise ValueError("首条输出已经低于阈值，无法夹逼交点。")
    previous_value = float(maximum_moisture[index - 1])
    current_value = float(maximum_moisture[index])
    if previous_value < threshold:
        raise ValueError("首次达标记录之前的数值未形成阈值夹逼。")
    crossing = float(times_h[index - 1]) + (
        (previous_value - threshold) / (previous_value - current_value)
    ) * float(times_h[index] - times_h[index - 1])
    return crossing, float(times_h[index])


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    boundary_time, room_temperature, room_moisture = read_boundary()
    times, radius_cm, moisture = read_result()
    maximum_moisture = moisture.max(axis=1)
    center_moisture = moisture[:, 0]
    surface_moisture = moisture[:, -1]
    threshold_index = int(np.flatnonzero(maximum_moisture < CRITICAL_MOISTURE)[0])
    continuous_threshold_time, discrete_threshold_time = threshold_times_from_output(
        times, maximum_moisture
    )

    # 1. 附件一原始边界
    figure, axis_temperature = plt.subplots(figsize=(7.2, 4.2))
    axis_moisture = axis_temperature.twinx()
    axis_temperature.plot(boundary_time, room_temperature, color="#1f77b4", label="烘房温度")
    axis_moisture.plot(boundary_time, room_moisture, color="#d62728", label="烘房水分浓度")
    axis_temperature.set(xlabel="时间 / h", ylabel="温度 / ℃", title="附件一烘房边界数据")
    axis_moisture.set_ylabel("水分浓度 / kg·kg$^{-1}$")
    axis_temperature.grid(alpha=0.25)
    save_figure(figure, "烘房边界原始数据")

    # 2. 最后 1 h 稳定段
    stable = boundary_time >= boundary_time[-1] - 1.0
    figure, axis_temperature = plt.subplots(figsize=(7.2, 4.2))
    axis_moisture = axis_temperature.twinx()
    axis_temperature.plot(boundary_time[stable], room_temperature[stable], "o-", ms=2.5,
                           color="#1f77b4", label="温度")
    axis_moisture.plot(boundary_time[stable], room_moisture[stable], "o-", ms=2.5,
                       color="#d62728", label="水分浓度")
    axis_temperature.axhline(room_temperature[stable].mean(), ls="--", color="#1f77b4", alpha=0.7)
    axis_moisture.axhline(room_moisture[stable].mean(), ls="--", color="#d62728", alpha=0.7)
    axis_temperature.set(xlabel="时间 / h", ylabel="温度 / ℃", title="附件一最后 1 h 稳定段")
    axis_moisture.set_ylabel("水分浓度 / kg·kg$^{-1}$")
    axis_temperature.grid(alpha=0.25)
    save_figure(figure, "烘房边界稳定段")

    # 3. 4 h 后边界平台与末点敏感性对照
    stable_temperature = room_temperature[stable].mean()
    stable_moisture = room_moisture[stable].mean()
    extension_time = np.linspace(4.0, max(24.0, times[-1]), 300)
    figure, axis_temperature = plt.subplots(figsize=(7.2, 4.2))
    axis_moisture = axis_temperature.twinx()
    axis_temperature.plot(extension_time, np.full_like(extension_time, stable_temperature),
                          color="#1f77b4", label="最后 1 h 均值")
    axis_temperature.plot(extension_time, np.full_like(extension_time, room_temperature[-1]),
                          color="#2ca02c", ls="--", label="最后采样点")
    axis_moisture.plot(extension_time, np.full_like(extension_time, stable_moisture),
                      color="#d62728", label="最后 1 h 均值")
    axis_moisture.plot(extension_time, np.full_like(extension_time, room_moisture[-1]),
                      color="#9467bd", ls="--", label="最后采样点")
    axis_temperature.set(xlabel="时间 / h", ylabel="温度 / ℃", title="4 h 后边界延拓情景")
    axis_moisture.set_ylabel("水分浓度 / kg·kg$^{-1}$")
    axis_temperature.grid(alpha=0.25)
    axis_temperature.legend(loc="center right", fontsize=9)
    save_figure(figure, "烘房边界延拓对比")

    # 4. 每隔 6 h 的径向剖面
    # 只选取少量具有代表性的时刻，避免 90 余条曲线和图例相互遮挡。
    selected_hours = np.array(
        [6.0, 12.0, 24.0, 36.0, 48.0, 54.0, discrete_threshold_time]
    )
    selected_hours = selected_hours[selected_hours <= times[-1] + 1.0e-9]
    selected_indices = [int(np.argmin(np.abs(times - h))) for h in selected_hours]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for index in selected_indices:
        if index == threshold_index:
            label = f"{times[index]:.4f} h（首次离散达标）"
        else:
            label = f"{times[index]:.0f} h"
        axis.plot(radius_cm, moisture[index], label=label)
    axis.axhline(CRITICAL_MOISTURE, color="black", ls="--", lw=1.0, label="阈值 0.15")
    axis.set(xlabel="到药材中心的距离 / cm", ylabel="含水率 / kg·kg$^{-1}$", title="含水率径向剖面")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    save_figure(figure, "含水率径向剖面")

    # 5. 中心、表面含水率轨迹
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(times, center_moisture, label="中心", color="#d62728")
    axis.plot(times, surface_moisture, label="表面", color="#1f77b4")
    axis.axhline(CRITICAL_MOISTURE, color="black", ls="--", label="阈值 0.15")
    axis.axvline(
        continuous_threshold_time,
        color="#ff7f0e",
        ls="-.",
        label=f"线性交点估计 t*={continuous_threshold_time:.4f} h",
    )
    axis.axvline(
        discrete_threshold_time,
        color="#555555",
        ls=":",
        label=f"首次 60 s 达标 t60={discrete_threshold_time:.4f} h",
    )
    axis.set(xlabel="时间 / h", ylabel="含水率 / kg·kg$^{-1}$", title="中心与表面含水率演化")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "中心表面含水率轨迹")

    # 6. 阈值前后径向剖面对照
    before_index = max(threshold_index - 1, 0)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(radius_cm, moisture[before_index], "o-", ms=2.5,
              label=f"达标前 {times[before_index]:.2f} h")
    axis.plot(radius_cm, moisture[threshold_index], "o-", ms=2.5,
              label=f"首次达标 {times[threshold_index]:.2f} h")
    axis.axhline(CRITICAL_MOISTURE, color="black", ls="--", label="阈值 0.15")
    axis.set(xlabel="到药材中心的距离 / cm", ylabel="含水率 / kg·kg$^{-1}$", title="阈值前后径向剖面")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "阈值前后剖面")

    # 7. 全域最大含水率与阈值
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(times, maximum_moisture, color="#2c3e50")
    axis.axhline(CRITICAL_MOISTURE, color="#d62728", ls="--", label="阈值 0.15")
    axis.axvline(
        continuous_threshold_time,
        color="#ff7f0e",
        ls="-.",
        label=f"t*={continuous_threshold_time:.4f} h",
    )
    axis.axvline(
        discrete_threshold_time,
        color="#555555",
        ls=":",
        label=f"t60={discrete_threshold_time:.4f} h",
    )
    axis.set(xlabel="时间 / h", ylabel="全域最大含水率 / kg·kg$^{-1}$", title="全域最大含水率下降曲线")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, "全域最大含水率下降")

    # 8. 含水率时空热图
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    image = axis.imshow(
        moisture.T, origin="lower", aspect="auto",
        extent=[times[0], times[-1], radius_cm[0], radius_cm[-1]],
        cmap="YlOrRd", vmin=CRITICAL_MOISTURE, vmax=float(moisture.max()),
    )
    figure.colorbar(image, ax=axis, label="含水率 / kg·kg$^{-1}$")
    axis.axvline(discrete_threshold_time, color="black", ls="--", lw=1.0)
    axis.set(xlabel="时间 / h", ylabel="到药材中心的距离 / cm", title="含水率时空分布")
    save_figure(figure, "含水率时空分布")

    # 9. 阈值交点局部放大
    left = max(threshold_index - 20, 0)
    right = min(threshold_index + 20, len(times) - 1)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(times[left:right + 1], maximum_moisture[left:right + 1], "o-", ms=3,
              color="#2c3e50")
    axis.axhline(CRITICAL_MOISTURE, color="#d62728", ls="--")
    axis.axvline(
        continuous_threshold_time,
        color="#ff7f0e",
        ls="-.",
        label=f"线性交点估计 t*={continuous_threshold_time:.4f} h",
    )
    axis.axvline(
        discrete_threshold_time,
        color="#555555",
        ls=":",
        label=f"首次 60 s 达标 t60={discrete_threshold_time:.4f} h",
    )
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.yaxis.set_major_formatter(formatter)
    axis.set(xlabel="时间 / h", ylabel="全域最大含水率 / kg·kg$^{-1}$", title="达标时间阈值判定局部图")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, "达标时间阈值判定")

    # 10. 已复算边界情景的达标时间比较
    if DIAGNOSTICS_FILE.exists():
        with DIAGNOSTICS_FILE.open("r", encoding="utf-8") as stream:
            diagnostics = json.load(stream)
        cases = [diagnostics["baseline"], *diagnostics["boundary_scenarios"]]
        labels = [case["label"] for case in cases]
        values = [case["continuous_threshold_time_h"] for case in cases]
        offsets_minutes = [(value - values[0]) * 60.0 for value in values]
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        positions = np.arange(len(cases))
        colors = ["#2c3e50", *["#1f77b4"] * (len(cases) - 1)]
        bars = axis.barh(positions, offsets_minutes, color=colors)
        axis.axvline(0.0, color="black", lw=0.8)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set(
            xlabel="相对基准情景的 t* 变化 / min",
            title="长期边界情景敏感性（非置信区间）",
        )
        axis.set_xlim(min(offsets_minutes) - 4.0, max(offsets_minutes) + 3.0)
        axis.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, offsets_minutes):
            if abs(value) >= 3.0:
                axis.text(
                    value / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.2f} min",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
            else:
                axis.text(
                    value + (0.35 if value >= 0 else -0.35),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.2f} min",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=8,
                )
        save_figure(figure, "长期边界情景敏感性")

    print(f"已生成 {len(list(FIGURE_DIR.glob('*.png')))} 张 PNG 图：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
