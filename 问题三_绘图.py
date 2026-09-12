"""问题三论文配图：长期边界、达标证据、拖尾机制与稳健性。

所有数据均来自附件和问题三求解诊断；脚本只导出 600 dpi PNG。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from openpyxl import load_workbook

import 问题3_求解 as problem3_solver


NATURE_FIGURE_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(NATURE_FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NATURE_FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_ROOT = Path(__file__).resolve().parent
BOUNDARY_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
RESULT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result3.xlsx"
DIAGNOSTICS_FILE = PROJECT_ROOT / "附件" / "附件3" / "result3_diagnostics.json"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题三"

CRITICAL_MOISTURE = 0.15
INITIAL_MOISTURE = 2.55
EXPORT_DPI = 600
PAD_INCHES = 0.06
FIGURE_FORMATS = ("png",)

CJK_FONT_CANDIDATES = (
    "Microsoft YaHei", "DengXian", "Source Han Sans SC", "Arial Unicode MS",
    "SimSun", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
)
INSTALLED_FONTS = {font.name for font in font_manager.fontManager.ttflist}
AVAILABLE_CJK_FONTS = [name for name in CJK_FONT_CANDIDATES if name in INSTALLED_FONTS]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": AVAILABLE_CJK_FONTS + ["Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "figure.dpi": 120,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

REFERENCE_PALETTE = ("#44757A", "#452A3D", "#D44C3C", "#EED5B7")
COLOR_TEAL, COLOR_PLUM, COLOR_CORAL, COLOR_SAND = REFERENCE_PALETTE
COLOR_DARK = "#51474D"

NEW_FIGURE_BASES = (
    "图1_长期边界平台依据",
    "图2_全域达标时间判定",
    "图3_含水率时空演化",
    "图4_后期拖尾机制",
    "图5_边界与数值稳健性",
)


def read_boundary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workbook = load_workbook(BOUNDARY_FILE, data_only=True, read_only=True)
    rows = [row for row in workbook.active.iter_rows(min_row=2, values_only=True) if row[0] is not None]
    data = np.asarray(rows, dtype=float)
    return data[:, 0] / 3600.0, data[:, 1], data[:, 2]


def read_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workbook = load_workbook(RESULT_FILE, data_only=True, read_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    radii_cm = np.asarray(rows[0][1:], dtype=float)
    times_h = np.asarray([row[0] for row in rows[1:]], dtype=float) / 3600.0
    moisture = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    return times_h, radii_cm, moisture


def threshold_times_from_output(
    times_h: np.ndarray,
    maximum_moisture: np.ndarray,
    threshold: float = CRITICAL_MOISTURE,
) -> tuple[float, float]:
    indices = np.flatnonzero(maximum_moisture < threshold)
    if len(indices) == 0 or indices[0] == 0:
        raise ValueError("输出记录没有形成有效的阈值夹逼。")
    index = int(indices[0])
    y0, y1 = float(maximum_moisture[index - 1]), float(maximum_moisture[index])
    crossing = float(times_h[index - 1]) + (y0 - threshold) / (y0 - y1) * float(
        times_h[index] - times_h[index - 1]
    )
    return crossing, float(times_h[index])


def load_plot_data() -> dict[str, object]:
    boundary_times_h, room_temperature, room_moisture = read_boundary()
    output_times_h, radii_cm, output_moisture = read_result()
    with DIAGNOSTICS_FILE.open("r", encoding="utf-8") as stream:
        diagnostics = json.load(stream)
    times_h = np.concatenate(([0.0], output_times_h))
    moisture = np.vstack((np.full((1, output_moisture.shape[1]), INITIAL_MOISTURE), output_moisture))
    baseline = diagnostics["baseline"]
    return {
        "boundary_times_h": boundary_times_h,
        "room_temperature": room_temperature,
        "room_moisture": room_moisture,
        "times_h": times_h,
        "radii_cm": radii_cm,
        "moisture": moisture,
        "center_moisture": moisture[:, 0],
        "surface_moisture": moisture[:, -1],
        "maximum_moisture": moisture.max(axis=1),
        "continuous_threshold_time_h": float(baseline["continuous_threshold_time_h"]),
        "discrete_threshold_time_h": float(baseline["discrete_threshold_time_h"]),
        "diagnostics": diagnostics,
    }


def drying_phase_rates(data: dict[str, object]) -> tuple[float, float]:
    times = np.asarray(data["times_h"], dtype=float)
    center = np.asarray(data["center_moisture"], dtype=float)
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("结果时间轴必须严格递增。")
    target = float(data["continuous_threshold_time_h"])
    c12 = float(center[int(np.argmin(np.abs(times - 12.0)))])
    c36 = float(center[int(np.argmin(np.abs(times - 36.0)))])
    return (INITIAL_MOISTURE - c12) / 12.0, (c36 - CRITICAL_MOISTURE) / (target - 36.0)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.tick_params(length=3, width=0.7)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(COLOR_DARK)
        axis.spines[name].set_linewidth(0.75)


def _set_top_title(figure: plt.Figure, title: str, y: float = 0.97) -> None:
    figure.suptitle(title, x=0.5, y=y, fontsize=12, fontweight="bold")


def close_without_export(figure: plt.Figure) -> None:
    plt.close(figure)


def save_publication_figure(
    figure: plt.Figure, base_name: str, exclude_axes: list[plt.Axes] | None = None
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    with TemporaryDirectory(prefix="problem3-figure-qa-") as temporary_dir:
        require_matplotlib_panel_alignment(
            figure,
            json_out=Path(temporary_dir) / f"{base_name}.alignment.json",
            exclude_axes=exclude_axes or [],
            row_groups=getattr(figure, "_alignment_row_groups", None),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    figure.savefig(
        FIGURE_DIR / f"{base_name}.png", dpi=EXPORT_DPI, bbox_inches="tight",
        pad_inches=PAD_INCHES, facecolor="white",
    )
    plt.close(figure)


def plot_boundary_plateau(data: dict[str, object]) -> plt.Figure:
    times = np.asarray(data["boundary_times_h"])
    temperature = np.asarray(data["room_temperature"])
    moisture = np.asarray(data["room_moisture"])
    stable = times >= times[-1] - 1.0
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    series = ((temperature, "温度 / ℃", COLOR_TEAL), (moisture, "环境含水率 / kg·kg$^{-1}$", COLOR_CORAL))
    for axis, (values, ylabel, color) in zip(axes, series):
        axis.plot(times[stable], values[stable], "o", ms=2.6, color=color, alpha=0.75)
        axis.axhline(values[stable].mean(), color=COLOR_PLUM, lw=1.5)
        axis.set(xlabel="时间 / h", ylabel=ylabel)
        axis.xaxis.set_major_locator(mticker.MaxNLocator(5))
        _style_axis(axis)
    axes[0].set_title("末 1 h 温度稳定段", pad=7)
    axes[1].set_title("末 1 h 水分边界稳定段", pad=7)
    figure.text(0.5, 0.845, "圆点为实测值，实线为末 1 h 均值", ha="center", color=COLOR_DARK, fontsize=7.3)
    _set_top_title(figure, "长期边界采用末 1 h 稳定平台", y=0.985)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.73, wspace=0.34)
    return figure


def plot_threshold_evidence(data: dict[str, object]) -> plt.Figure:
    times = np.asarray(data["times_h"])
    center = np.asarray(data["center_moisture"])
    surface = np.asarray(data["surface_moisture"])
    target = float(data["continuous_threshold_time_h"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axes[0].plot(times, center, color=COLOR_TEAL, lw=1.7, label="轴心（全域最大值）")
    axes[0].plot(times, surface, color=COLOR_CORAL, lw=1.35, label="表面")
    axes[0].axhline(CRITICAL_MOISTURE, color=COLOR_DARK, ls="--", lw=1.0, label="达标阈值")
    axes[0].set(xlabel="时间 / h", ylabel="含水率 / kg·kg$^{-1}$", title="全过程")
    zoom = (times >= target - 0.22) & (times <= target + 0.18)
    axes[1].plot(times[zoom], center[zoom], "o-", ms=2.3, color=COLOR_TEAL, lw=1.25)
    axes[1].axhline(CRITICAL_MOISTURE, color=COLOR_DARK, ls="--", lw=1.0)
    axes[1].axvline(target, color=COLOR_CORAL, lw=1.35)
    axes[1].annotate(
        f"$t_*= {target:.4f}$ h", xy=(target, CRITICAL_MOISTURE),
        xytext=(target - 0.19, CRITICAL_MOISTURE + 0.00008),
        arrowprops={"arrowstyle": "-", "color": COLOR_CORAL, "lw": 0.8},
        color=COLOR_CORAL, fontsize=7.5,
    )
    axes[1].set(xlabel="时间 / h", ylabel="轴心含水率 / kg·kg$^{-1}$", title="阈值附近")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for axis in axes:
        _style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=3)
    _set_top_title(figure, "轴心控制全域达标时间：57.52 h", y=0.985)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.70, wspace=0.34)
    return figure


def plot_field_evolution(data: dict[str, object]) -> plt.Figure:
    times = np.asarray(data["times_h"])
    radii = np.asarray(data["radii_cm"])
    moisture = np.asarray(data["moisture"])
    discrete = float(data["discrete_threshold_time_h"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    mesh = axes[0].pcolormesh(times, radii, moisture.T, shading="auto", cmap="viridis")
    axes[0].axvline(discrete, color="white", ls="--", lw=1.0)
    axes[0].set(xlabel="时间 / h", ylabel="径向位置 / cm", title="时空分布")
    color_axis = axes[0].inset_axes([1.025, 0.02, 0.045, 0.96])
    figure.colorbar(mesh, cax=color_axis)
    selected = (0.0, 6.0, 12.0, 24.0, 36.0, discrete)
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.90, len(selected)))
    for hour, color in zip(selected, colors):
        index = int(np.argmin(np.abs(times - hour)))
        label = f"{times[index]:.1f} h" if hour else "0 h"
        axes[1].plot(radii, moisture[index], color=color, lw=1.45, label=label)
    axes[1].axhline(CRITICAL_MOISTURE, color=COLOR_DARK, ls="--", lw=0.9)
    axes[1].set(xlabel="径向位置 / cm", ylabel="含水率 / kg·kg$^{-1}$", title="代表时刻径向剖面")
    for axis in axes:
        _style_axis(axis)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=6)
    _set_top_title(figure, "含水率由表面向轴心逐步衰减", y=0.985)
    figure.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.69, wspace=0.40)
    figure._alignment_row_groups = [["a", "b"]]
    figure._extra_qa_axes = [color_axis]
    return figure


def plot_drying_tail(data: dict[str, object]) -> plt.Figure:
    times = np.asarray(data["times_h"])
    center = np.asarray(data["center_moisture"])
    diagnostics = data["diagnostics"]
    plateau_temperature = float(diagnostics["baseline"]["plateau_temperature_c"])
    diffusivity = problem3_solver.moisture_diffusivity(center, np.full_like(center, plateau_temperature))
    early_rate, late_rate = drying_phase_rates(data)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axes[0].plot(times, center, color=COLOR_TEAL, lw=1.7)
    axes[0].axhline(CRITICAL_MOISTURE, color=COLOR_DARK, ls="--", lw=0.9)
    axes[0].axvspan(0.0, 12.0, color=COLOR_SAND, alpha=0.48)
    axes[0].axvspan(36.0, times[-1], color=COLOR_CORAL, alpha=0.10)
    axes[0].set(xlabel="时间 / h", ylabel="轴心含水率 / kg·kg$^{-1}$", title="两阶段干燥过程")
    axes[1].semilogy(times, diffusivity / diffusivity[0], color=COLOR_PLUM, lw=1.7)
    axes[1].set(xlabel="时间 / h", ylabel="相对扩散系数 $D/D_0$", title="低含水率下扩散能力衰减")
    axes[1].text(
        0.97, 0.93, f"初期 {early_rate:.4f}\n后期 {late_rate:.4f}\n速率相差 >100 倍",
        transform=axes[1].transAxes, ha="right", va="top", color=COLOR_DARK,
        fontsize=7.4, linespacing=1.35,
    )
    for axis in axes:
        _style_axis(axis)
    _set_top_title(figure, "扩散系数退化导致后期拖尾", y=0.985)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.74, wspace=0.34)
    return figure


def _lollipop(axis: plt.Axes, labels: list[str], values: np.ndarray, xlabel: str) -> None:
    positions = np.arange(len(values))
    colors = [COLOR_CORAL if value > 0 else COLOR_TEAL for value in values]
    axis.hlines(positions, 0.0, values, color=colors, lw=2.0)
    axis.scatter(values, positions, color=colors, s=27, zorder=3)
    axis.axvline(0.0, color=COLOR_DARK, lw=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    span = max(np.max(np.abs(values)), 1.0)
    axis.set_xlim(
        min(float(np.min(values)), 0.0) - 0.08 * span,
        max(float(np.max(values)), 0.0) + 0.18 * span,
    )
    for y, value in zip(positions, values):
        axis.text(value + span * 0.025, y, f"{value:+.1f}", va="center", ha="left", fontsize=7.2)
    _style_axis(axis)


def plot_robustness(data: dict[str, object]) -> plt.Figure:
    diagnostics = data["diagnostics"]
    baseline = float(diagnostics["baseline"]["continuous_threshold_time_h"])
    boundary_cases = diagnostics["boundary_scenarios"]
    boundary_labels = ["末 30 min", "最后采样点", "温度 -1σ", "温度 +1σ", "水分 -1σ", "水分 +1σ"]
    boundary_offsets = 60.0 * np.asarray([case["continuous_threshold_time_h"] - baseline for case in boundary_cases])
    refinements = diagnostics["numerical_refinement"]
    numerical_labels = ["空间加密", "时间加密", "端面暴露"]
    numerical_offsets = np.asarray([
        3600.0 * (refinements[0]["continuous_threshold_time_h"] - baseline),
        3600.0 * (refinements[1]["continuous_threshold_time_h"] - baseline),
        -7.6,
    ])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
    _lollipop(axes[0], boundary_labels, boundary_offsets, "相对基准达标时间变化 / min")
    _lollipop(axes[1], numerical_labels, numerical_offsets, "达标时间变化 / s")
    axes[0].set_title("长期边界情景", pad=7)
    axes[1].set_title("离散与端面检验", pad=7)
    _set_top_title(figure, "主要结论对边界扰动与数值设置稳定", y=0.985)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.74, wspace=0.58)
    return figure


def remove_legacy_outputs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    expected = {f"{name}.png" for name in NEW_FIGURE_BASES}
    for target in FIGURE_DIR.iterdir():
        if target.is_file() and target.suffix.lower() in {".png", ".pdf", ".svg", ".tif", ".tiff"} and target.name not in expected:
            if target.resolve().parent != FIGURE_DIR.resolve():
                raise RuntimeError("拒绝删除问题三图片目录之外的文件。")
            target.unlink()


def export_all(data: dict[str, object]) -> None:
    remove_legacy_outputs()
    builders = (plot_boundary_plateau, plot_threshold_evidence, plot_field_evolution, plot_drying_tail, plot_robustness)
    for builder, base_name in zip(builders, NEW_FIGURE_BASES):
        figure = builder(data)
        save_publication_figure(figure, base_name, getattr(figure, "_extra_qa_axes", None))


def main() -> None:
    export_all(load_plot_data())
    print(f"问题三绘图完成，共生成 {len(NEW_FIGURE_BASES)} 张 600 dpi PNG：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
