"""问题一绘图脚本。

运行位置：项目根目录
输入：附件/附件1.xlsx、附件/附件3/result1.xlsx
输出：figures/问题一/ 下的 PNG 图片

脚本不把数据写死在代码中，所有曲线和热力图均由当前结果文件重新读取生成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


# ============================================================
# 一、输入输出路径与外部绘图工具
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ATTACHMENT_ONE = PROJECT_ROOT / "附件" / "附件1.xlsx"
RESULT_ONE = PROJECT_ROOT / "附件" / "附件3" / "result1.xlsx"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题一"

# 统一导出工具（多格式、固定尺寸、灰度预览），从技能目录动态载入
FIGURE_TOOL_DIR = Path(
    r"C:\Users\Lenovo\.codex\skills\math-modeling-skill\tools\figure\scripts"
)
sys.path.insert(0, str(FIGURE_TOOL_DIR))
from export_figure import export_figure  # noqa: E402


# ============================================================
# 二、全局绘图风格与配色
# ============================================================

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

# 径向剖面图的取样时刻，单位 s
SNAPSHOT_TIMES = [100, 600, 1200, 1800]

# 论文表格取值时刻，单位 s（当前绘图脚本未使用，供论文制表参考）
PAPER_TABLE_TIMES = [100, 300, 600, 900, 1200, 1500, 1800]

# 剖面曲线配色，与 SNAPSHOT_TIMES 一一对应
PROFILE_COLORS = ["#31688e", "#35b779", "#fde725", "#d73027"]

# 通用强调色
COLOR_COOL = "#31688e"    # 冷色强调（蓝）
COLOR_WARM = "#d73027"    # 暖色强调（红）
COLOR_BAR = "#7e9fbe"     # 柱状图填充色

# 各类图的成图尺寸（英寸），同时也是导出尺寸
RAW_SIZE = (6.4, 4.2)         # 附件一曲线图
SAMPLING_SIZE = (5.0, 3.2)    # 采样间隔柱状图
PROFILE_SIZE = (6.4, 4.2)     # 径向剖面图
HEATMAP_SIZE = (6.4, 4.2)     # 时空热力图
WIDE_SIZE = (7.0, 3.2)        # 并排双子图

# PNG 导出分辨率与裁边留白（英寸）。留白显式取 0.1：export_figure 的灰度预览
# 分支会用 rcParams 默认的 0.1 重存一次 PNG，取同一个值才能与既往图片逐字节一致。
EXPORT_DPI = 300
PAD_INCHES = 0.1


# ============================================================
# 三、绘图基础件
# ============================================================


def new_axes(
    title: str,
    xlabel: str,
    ylabel: str,
    size: tuple[float, float] = PROFILE_SIZE,
    grid_axis: str | None = "both",
):
    """新建单轴图，统一设置标题、轴标签和网格。

    grid_axis 为 "both"/"x"/"y" 时按该方向加网格，为 None 时不加网格。
    """
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.25)
    return fig, ax


def save_figure(
    fig,
    filename: str,
    size: tuple[float, float] = PROFILE_SIZE,
) -> None:
    """按统一尺寸导出 PNG，并关闭图形。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(
        fig,
        str(FIGURE_DIR / filename),
        formats=["png"],
        dpi=EXPORT_DPI,
        size_inches=size,
        pad_inches=PAD_INCHES,
    )
    plt.close(fig)


def snapshot_index(times: np.ndarray, t: float) -> int:
    """定位结果文件中等于给定时刻的行号。"""
    return int(np.where(times == t)[0][0])


def plot_profiles(ax, times: np.ndarray, radii: np.ndarray, field: np.ndarray) -> None:
    """在同一坐标轴上按取样时刻叠加径向剖面曲线。"""
    for color, t in zip(PROFILE_COLORS, SNAPSHOT_TIMES):
        ax.plot(
            radii,
            field[snapshot_index(times, t)],
            color=color,
            linewidth=1.8,
            label=f"{t} s",
        )
    ax.legend(frameon=False, ncol=2)


# ============================================================
# 四、数据读取
# ============================================================


def read_boundary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取附件一的时间、烘房温度和烘房水分浓度。"""
    workbook = load_workbook(ATTACHMENT_ONE, data_only=True, read_only=True)
    worksheet = workbook.active
    data = np.asarray(
        [
            row for row in worksheet.iter_rows(min_row=2, values_only=True)
            if row[0] is not None
        ],
        dtype=float,
    )
    return data[:, 0], data[:, 1], data[:, 2]


def read_problem_one_result() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """读取 result1.xlsx 中的时间、半径、温度场和含水率场。

    表头首列是时间标记，其余各列给出节点半径；两个工作表的半径列相同，
    因此只从第一个工作表读取半径。
    """
    workbook = load_workbook(RESULT_ONE, data_only=True, read_only=True)
    temperature_sheet, moisture_sheet = workbook.worksheets[:2]

    def read_sheet(sheet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = list(sheet.iter_rows(min_row=2, values_only=True))
        times = np.asarray([row[0] for row in rows], dtype=float)
        header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        radii = np.asarray(list(header)[1:], dtype=float)
        field = np.asarray([row[1:] for row in rows], dtype=float)
        return times, radii, field

    times, radii, temperature_field = read_sheet(temperature_sheet)
    _, _, moisture_field = read_sheet(moisture_sheet)
    return times, radii, temperature_field, moisture_field


# ============================================================
# 五、成图
# ============================================================


def plot_raw_data(
    boundary_times: np.ndarray,
    drying_temperature: np.ndarray,
    drying_moisture: np.ndarray,
) -> None:
    """绘制附件一的两条边界曲线及其采样间隔分布。"""
    fig, ax = new_axes(
        "烘房温度边界随时间的变化",
        "时间 / h",
        "烘房温度 / °C",
        size=RAW_SIZE,
    )
    ax.plot(boundary_times / 3600, drying_temperature, color=COLOR_COOL, linewidth=1.8)
    save_figure(fig, "raw_q1_烘房温度", size=RAW_SIZE)

    fig, ax = new_axes(
        "烘房水分浓度边界随时间的变化",
        "时间 / h",
        "烘房水分浓度 / kg/kg",
        size=RAW_SIZE,
    )
    ax.plot(boundary_times / 3600, drying_moisture, color=COLOR_WARM, linewidth=1.8)
    save_figure(fig, "raw_q1_烘房水分浓度", size=RAW_SIZE)

    fig, ax = new_axes(
        "附件一边界数据的采样间隔",
        "采样间隔 / s",
        "出现次数",
        size=SAMPLING_SIZE,
        grid_axis="y",
    )
    intervals = np.diff(boundary_times)
    unique_intervals, counts = np.unique(intervals, return_counts=True)
    ax.bar(
        unique_intervals,
        counts,
        width=max(1.0, float(unique_intervals[0]) * 0.15),
        color=COLOR_BAR,
    )
    ax.set_xticks(unique_intervals)
    save_figure(fig, "raw_q1_边界采样间隔", size=SAMPLING_SIZE)


def plot_process(
    times: np.ndarray,
    radii: np.ndarray,
    temperature_field: np.ndarray,
    moisture_field: np.ndarray,
) -> None:
    """绘制径向剖面图与中心—表面轨迹图。"""
    fig, ax = new_axes(
        "不同时间的药材径向温度剖面",
        "到药材中心的距离 / cm",
        "温度 / °C",
        size=PROFILE_SIZE,
    )
    plot_profiles(ax, times, radii, temperature_field)
    save_figure(fig, "process_q1_温度径向剖面", size=PROFILE_SIZE)

    fig, ax = new_axes(
        "不同时间的药材径向含水率剖面",
        "到药材中心的距离 / cm",
        "干基含水率 / kg/kg",
        size=PROFILE_SIZE,
    )
    plot_profiles(ax, times, radii, moisture_field)
    save_figure(fig, "process_q1_水分径向剖面", size=PROFILE_SIZE)

    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE, constrained_layout=True)
    for ax, field, ylabel, title in [
        (axes[0], temperature_field, "温度 / °C", "温度中心—表面轨迹"),
        (axes[1], moisture_field, "干基含水率 / kg/kg", "含水率中心—表面轨迹"),
    ]:
        ax.plot(times, field[:, 0], color=COLOR_COOL, label="中心", linewidth=1.6)
        ax.plot(times, field[:, -1], color=COLOR_WARM, label="表面", linewidth=1.6)
        ax.set_xlabel("时间 / s")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    save_figure(fig, "process_q1_中心表面轨迹", size=WIDE_SIZE)


def plot_results(
    times: np.ndarray,
    radii: np.ndarray,
    temperature_field: np.ndarray,
    moisture_field: np.ndarray,
) -> None:
    """绘制温度场、含水率场的时空热力图及两者的空间非均匀性。"""
    fig, ax = new_axes(
        "问题一温度场时空分布",
        "到药材中心的距离 / cm",
        "时间 / s",
        size=HEATMAP_SIZE,
        grid_axis=None,
    )
    mesh = ax.pcolormesh(radii, times, temperature_field, shading="auto", cmap="magma")
    fig.colorbar(mesh, ax=ax, label="温度 / °C")
    save_figure(fig, "result_q1_温度场热力图", size=HEATMAP_SIZE)

    fig, ax = new_axes(
        "问题一含水率场时空分布",
        "到药材中心的距离 / cm",
        "时间 / s",
        size=HEATMAP_SIZE,
        grid_axis=None,
    )
    mesh = ax.pcolormesh(radii, times, moisture_field, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="干基含水率 / kg/kg")
    save_figure(fig, "result_q1_水分场热力图", size=HEATMAP_SIZE)

    fig, axes = plt.subplots(1, 2, figsize=WIDE_SIZE, constrained_layout=True)
    for ax, curve, color, ylabel, title in [
        (
            axes[0],
            temperature_field[:, -1] - temperature_field[:, 0],
            COLOR_WARM,
            "表面－中心温差 / °C",
            "温度空间非均匀性",
        ),
        (
            axes[1],
            moisture_field[:, 0] - moisture_field[:, -1],
            COLOR_COOL,
            "中心－表面含水率差 / kg/kg",
            "含水率空间非均匀性",
        ),
    ]:
        ax.plot(times, curve, color=color, linewidth=1.8)
        ax.set_xlabel("时间 / s")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    save_figure(fig, "result_q1_中心表面差值", size=WIDE_SIZE)


def main() -> None:
    boundary_times, drying_temperature, drying_moisture = read_boundary_data()
    times, radii, temperature_field, moisture_field = read_problem_one_result()
    plot_raw_data(boundary_times, drying_temperature, drying_moisture)
    plot_process(times, radii, temperature_field, moisture_field)
    plot_results(times, radii, temperature_field, moisture_field)
    print(f"问题一绘图完成，共输出到：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
