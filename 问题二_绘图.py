"""问题二绘图脚本。

输入：附件/附件1.xlsx、附件/附件3/result2.xlsx
输出：figures/问题二/ 下的 PNG 图像

绘图脚本不写死数值，所有图均从当前附件和 result2.xlsx 重新读取生成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parent
BOUNDARY_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
RESULT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result2.xlsx"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题二"

FIGURE_TOOL_DIR = Path(
    r"C:\Users\Lenovo\.codex\skills\math-modeling-skill\tools\figure\scripts"
)
sys.path.insert(0, str(FIGURE_TOOL_DIR))
from export_figure import export_figure  # noqa: E402


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
})


PLOT_TIMES = [1800, 3600, 5400, 7200, 9000, 10800]
PROFILE_TIMES = [1800, 5400, 9000, 10800]
PROFILE_COLORS = ["#31688e", "#35b779", "#fde725", "#d73027"]


def read_boundary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取附件一中的边界时间、温度和水分浓度。"""
    workbook = load_workbook(BOUNDARY_FILE, data_only=True, read_only=True)
    worksheet = workbook.active
    rows = [
        row for row in worksheet.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def read_result() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 result2.xlsx 的温度场和含水率场。"""
    workbook = load_workbook(RESULT_FILE, data_only=True, read_only=True)
    temperature_sheet, moisture_sheet = workbook.worksheets[:2]

    def read_sheet(worksheet):
        rows = list(worksheet.iter_rows(values_only=True))
        times = np.asarray([row[0] for row in rows[1:]], dtype=float)
        radius_cm = np.asarray(rows[0][1:], dtype=float)
        field = np.asarray([row[1:] for row in rows[1:]], dtype=float)
        return times, radius_cm, field

    times, radius_cm, temperature_field = read_sheet(temperature_sheet)
    _, _, moisture_field = read_sheet(moisture_sheet)
    return times, radius_cm, temperature_field, moisture_field


def export_png(figure, filename: str, size_inches: tuple[float, float]) -> None:
    """统一导出 300 DPI PNG；按用户要求不生成 SVG、PDF 和灰度副本。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(
        figure,
        str(FIGURE_DIR / filename),
        formats=["png"],
        dpi=300,
        size_inches=size_inches,
        grayscale_preview=False,
    )
    plt.close(figure)


def draw_raw_data(boundary_time, room_temperature, room_moisture) -> None:
    """原始数据图：展示问题二 0–3 h 的边界变化。"""
    mask = boundary_time <= 10800
    time_hours = boundary_time[mask] / 3600.0
    temperature = room_temperature[mask]
    moisture = room_moisture[mask]

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    axes[0].plot(time_hours, temperature, color="#d73027", linewidth=1.8)
    axes[0].set_xlabel("时间 / h")
    axes[0].set_ylabel("烘房温度 / °C")
    axes[0].set_title("0–3 h 烘房温度边界")
    axes[1].plot(time_hours, moisture, color="#31688e", linewidth=1.8)
    axes[1].set_xlabel("时间 / h")
    axes[1].set_ylabel("烘房水分浓度 / kg/kg")
    axes[1].set_title("0–3 h 烘房水分边界")
    for axis in axes:
        axis.grid(alpha=0.25)
    export_png(figure, "raw_q2_烘房边界", (7.0, 3.1))

    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    intervals = np.diff(boundary_time[mask])
    values, counts = np.unique(intervals, return_counts=True)
    axis.bar(values, counts, width=max(1.0, float(values[0]) * 0.15), color="#7e9fbe")
    axis.set_xlabel("采样间隔 / s")
    axis.set_ylabel("出现次数")
    axis.set_title("问题二边界数据采样间隔")
    axis.set_xticks(values)
    axis.grid(axis="y", alpha=0.25)
    export_png(figure, "raw_q2_边界采样间隔", (5.2, 3.2))

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    axes[0].plot(time_hours[1:], np.diff(temperature) / np.diff(time_hours),
                 color="#d73027", linewidth=1.4)
    axes[0].set_xlabel("时间 / h")
    axes[0].set_ylabel("温度变化率 / °C/h")
    axes[0].set_title("烘房升温速率")
    axes[1].plot(time_hours[1:], np.diff(moisture) / np.diff(time_hours),
                 color="#31688e", linewidth=1.4)
    axes[1].set_xlabel("时间 / h")
    axes[1].set_ylabel("水分浓度变化率 / (kg/kg)/h")
    axes[1].set_title("烘房水分浓度变化率")
    for axis in axes:
        axis.grid(alpha=0.25)
    export_png(figure, "raw_q2_边界变化率", (7.0, 3.1))


def draw_process_profiles(times, radius_cm, temperature_field, moisture_field) -> None:
    """过程图：展示若干时刻的径向剖面和中心—表面轨迹。"""
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for color, seconds in zip(PROFILE_COLORS, PROFILE_TIMES):
        index = int(np.argmin(np.abs(times - seconds)))
        axis.plot(radius_cm, temperature_field[index], color=color,
                  linewidth=1.8, label=f"{seconds / 3600:.1f} h")
    axis.set_xlabel("到药材中心的距离 / cm")
    axis.set_ylabel("温度 / °C")
    axis.set_title("问题二变物性模型的径向温度剖面")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.25)
    export_png(figure, "process_q2_温度径向剖面", (6.4, 4.2))

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for color, seconds in zip(PROFILE_COLORS, PROFILE_TIMES):
        index = int(np.argmin(np.abs(times - seconds)))
        axis.plot(radius_cm, moisture_field[index], color=color,
                  linewidth=1.8, label=f"{seconds / 3600:.1f} h")
    axis.set_xlabel("到药材中心的距离 / cm")
    axis.set_ylabel("干基含水率 / kg/kg")
    axis.set_title("问题二变物性模型的径向含水率剖面")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.25)
    export_png(figure, "process_q2_含水率径向剖面", (6.4, 4.2))

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    axes[0].plot(times / 3600.0, temperature_field[:, 0],
                 color="#31688e", linewidth=1.6, label="中心")
    axes[0].plot(times / 3600.0, temperature_field[:, -1],
                 color="#d73027", linewidth=1.6, label="表面")
    axes[0].set_xlabel("时间 / h")
    axes[0].set_ylabel("温度 / °C")
    axes[0].set_title("温度中心—表面轨迹")
    axes[1].plot(times / 3600.0, moisture_field[:, 0],
                 color="#31688e", linewidth=1.6, label="中心")
    axes[1].plot(times / 3600.0, moisture_field[:, -1],
                 color="#d73027", linewidth=1.6, label="表面")
    axes[1].set_xlabel("时间 / h")
    axes[1].set_ylabel("干基含水率 / kg/kg")
    axes[1].set_title("含水率中心—表面轨迹")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
    export_png(figure, "process_q2_中心表面轨迹", (7.0, 3.2))


def draw_final_results(times, radius_cm, temperature_field, moisture_field) -> None:
    """结果图：展示时空场和空间非均匀性。"""
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    image = axis.pcolormesh(
        radius_cm, times / 3600.0, temperature_field,
        shading="auto", cmap="magma",
    )
    axis.set_xlabel("到药材中心的距离 / cm")
    axis.set_ylabel("时间 / h")
    axis.set_title("问题二温度场时空分布")
    figure.colorbar(image, ax=axis, label="温度 / °C")
    export_png(figure, "result_q2_温度场热力图", (6.4, 4.2))

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    image = axis.pcolormesh(
        radius_cm, times / 3600.0, moisture_field,
        shading="auto", cmap="viridis",
    )
    axis.set_xlabel("到药材中心的距离 / cm")
    axis.set_ylabel("时间 / h")
    axis.set_title("问题二含水率场时空分布")
    figure.colorbar(image, ax=axis, label="干基含水率 / kg/kg")
    export_png(figure, "result_q2_含水率场热力图", (6.4, 4.2))

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    temperature_gap = temperature_field[:, -1] - temperature_field[:, 0]
    moisture_gap = moisture_field[:, 0] - moisture_field[:, -1]
    axes[0].plot(times / 3600.0, temperature_gap,
                 color="#d73027", linewidth=1.8)
    axes[0].set_xlabel("时间 / h")
    axes[0].set_ylabel("表面－中心温差 / °C")
    axes[0].set_title("温度空间非均匀性")
    axes[1].plot(times / 3600.0, moisture_gap,
                 color="#31688e", linewidth=1.8)
    axes[1].set_xlabel("时间 / h")
    axes[1].set_ylabel("中心－表面含水率差 / kg/kg")
    axes[1].set_title("含水率空间非均匀性")
    for axis in axes:
        axis.grid(alpha=0.25)
    export_png(figure, "result_q2_空间非均匀性", (7.0, 3.2))


def main() -> None:
    boundary_time, room_temperature, room_moisture = read_boundary()
    times, radius_cm, temperature_field, moisture_field = read_result()
    draw_raw_data(boundary_time, room_temperature, room_moisture)
    draw_process_profiles(times, radius_cm, temperature_field, moisture_field)
    draw_final_results(times, radius_cm, temperature_field, moisture_field)
    print(f"问题二绘图完成，共输出到：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
