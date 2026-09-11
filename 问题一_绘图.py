"""问题一绘图脚本。

输入：附件/附件1.xlsx、附件/附件3/result1.xlsx
输出：figures/问题一/ 下的 9 张 PNG 图片（中文文件名）

所有曲线和热力图都由结果文件现场读取生成，脚本内不写死任何数值。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


# 路径一律以脚本所在目录为基准，换电脑或换工作目录都不用改代码
PROJECT_ROOT = Path(__file__).resolve().parent
ATTACHMENT_ONE = PROJECT_ROOT / "附件" / "附件1.xlsx"
RESULT_ONE = PROJECT_ROOT / "附件" / "附件3" / "result1.xlsx"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题一"

# 中文字体候选，按顺序取本机第一个已安装的，换系统后不会出现方框字
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS",      # Windows
    "PingFang SC", "Heiti SC", "STHeiti",                           # macOS
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",  # Linux
]
INSTALLED_FONTS = {font.name for font in font_manager.fontManager.ttflist}
AVAILABLE_CJK_FONTS = [name for name in CJK_FONT_CANDIDATES if name in INSTALLED_FONTS]
if not AVAILABLE_CJK_FONTS:
    print("提示：本机未检测到中文字体，图中中文可能显示为方框，请先安装中文字体。")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": AVAILABLE_CJK_FONTS + ["DejaVu Sans"],
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

# 论文表格取值时刻，单位 s（绘图脚本未使用，供正文制表参考）
PAPER_TABLE_TIMES = [100, 300, 600, 900, 1200, 1500, 1800]

# 剖面曲线配色，与 SNAPSHOT_TIMES 一一对应
PROFILE_COLORS = ["#31688e", "#35b779", "#fde725", "#d73027"]

COLOR_COOL = "#31688e"    # 冷色强调（蓝）
COLOR_WARM = "#d73027"    # 暖色强调（红）
COLOR_BAR = "#7e9fbe"     # 柱状图填充色

# 各类图的成图尺寸（英寸），同时也是导出尺寸
RAW_SIZE = (6.4, 4.2)         # 附件一曲线图
SAMPLING_SIZE = (5.0, 3.2)    # 采样间隔柱状图
PROFILE_SIZE = (6.4, 4.2)     # 径向剖面图
HEATMAP_SIZE = (6.4, 4.2)     # 时空热力图
WIDE_SIZE = (7.0, 3.2)        # 并排双子图

EXPORT_DPI = 300              # PNG 导出分辨率
PAD_INCHES = 0.1              # 裁边后保留的留白，单位英寸


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
    """按固定物理尺寸导出 300 DPI PNG，并关闭图形。

    用 bbox_inches="tight" 裁掉多余留白，导出尺寸与 size 基本一致，
    图片可直接贴进论文，无需二次缩放。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*size)
    fig.savefig(
        FIGURE_DIR / filename,
        dpi=EXPORT_DPI,
        bbox_inches="tight",
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
    save_figure(fig, "图1_附件一烘房温度曲线.png", size=RAW_SIZE)

    fig, ax = new_axes(
        "烘房水分浓度边界随时间的变化",
        "时间 / h",
        "烘房水分浓度 / kg/kg",
        size=RAW_SIZE,
    )
    ax.plot(boundary_times / 3600, drying_moisture, color=COLOR_WARM, linewidth=1.8)
    save_figure(fig, "图2_附件一烘房水分浓度曲线.png", size=RAW_SIZE)

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
    save_figure(fig, "图3_附件一边界采样间隔.png", size=SAMPLING_SIZE)


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
    save_figure(fig, "图4_温度径向剖面.png", size=PROFILE_SIZE)

    fig, ax = new_axes(
        "不同时间的药材径向含水率剖面",
        "到药材中心的距离 / cm",
        "干基含水率 / kg/kg",
        size=PROFILE_SIZE,
    )
    plot_profiles(ax, times, radii, moisture_field)
    save_figure(fig, "图5_水分径向剖面.png", size=PROFILE_SIZE)

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
    save_figure(fig, "图6_中心与表面轨迹.png", size=WIDE_SIZE)


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
    save_figure(fig, "图7_温度场时空分布.png", size=HEATMAP_SIZE)

    fig, ax = new_axes(
        "问题一含水率场时空分布",
        "到药材中心的距离 / cm",
        "时间 / s",
        size=HEATMAP_SIZE,
        grid_axis=None,
    )
    mesh = ax.pcolormesh(radii, times, moisture_field, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="干基含水率 / kg/kg")
    save_figure(fig, "图8_水分场时空分布.png", size=HEATMAP_SIZE)

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
    save_figure(fig, "图9_空间非均匀性.png", size=WIDE_SIZE)


def main() -> None:
    boundary_times, drying_temperature, drying_moisture = read_boundary_data()
    times, radii, temperature_field, moisture_field = read_problem_one_result()
    plot_raw_data(boundary_times, drying_temperature, drying_moisture)
    plot_process(times, radii, temperature_field, moisture_field)
    plot_results(times, radii, temperature_field, moisture_field)
    print(f"问题一绘图完成，共输出到：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
