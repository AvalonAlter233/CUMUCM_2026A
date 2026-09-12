"""问题一论文配图：实测边界、热湿响应、场演化和数值验证。

数据源：附件一边界数据、result1.xlsx 以及问题一求解器的现场重算结果。
脚本不写入求解结果，只在新图全部成功导出后清理旧问题一 PNG。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from openpyxl import load_workbook

import 问题1_求解 as problem1_solver


NATURE_FIGURE_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(NATURE_FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NATURE_FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_ROOT = Path(__file__).resolve().parent
ATTACHMENT_ONE = PROJECT_ROOT / "附件" / "附件1.xlsx"
RESULT_ONE = PROJECT_ROOT / "附件" / "附件3" / "result1.xlsx"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题一"

CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "DengXian",
    "Source Han Serif SC",
    "Arial Unicode MS",
    "SimSun",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)
INSTALLED_FONTS = {font.name for font in font_manager.fontManager.ttflist}
AVAILABLE_CJK_FONTS = [
    name for name in CJK_FONT_CANDIDATES if name in INSTALLED_FONTS
]
if not AVAILABLE_CJK_FONTS:
    print("提示：本机未检测到中文字体，图中中文可能显示为方框。")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": AVAILABLE_CJK_FONTS + ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 120,
    }
)

SNAPSHOT_TIMES = (100.0, 600.0, 1200.0, 1800.0)
EXPORT_DPI = 600
PAD_INCHES = 0.06
FIGURE_FORMATS = ("png",)

REFERENCE_PALETTE = ("#44757A", "#452A3D", "#D44C3C", "#EED5B7")
COLOR_TEAL, COLOR_PLUM, COLOR_CORAL, COLOR_SAND = REFERENCE_PALETTE
COLOR_CENTER = COLOR_TEAL
COLOR_AVERAGE = COLOR_PLUM
COLOR_SURFACE = COLOR_CORAL
PROFILE_COLORS = (COLOR_TEAL, COLOR_PLUM, "#B88E63", COLOR_CORAL)

NEW_FIGURE_BASES = (
    "图1_实测边界条件",
    "图2_中心平均表面响应",
    "图3_热湿场时空演化",
    "图4_热湿径向剖面",
    "图5_数值验证",
)


def prepend_initial_state(
    times: np.ndarray, field: np.ndarray, initial_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """将已知的 t=0 均匀初始场加入时空场数据。"""
    if field.ndim != 2 or field.shape[0] != len(times):
        raise ValueError("时间轴与场数组的第一维不一致。")
    initial_row = np.full((1, field.shape[1]), float(initial_value))
    return np.concatenate(([0.0], times), axis=0), np.vstack((initial_row, field))


def max_radial_error(reference: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """计算每个时刻沿全部径向输出点的最大绝对差。"""
    if reference.shape != refined.shape or reference.ndim != 2:
        raise ValueError("待比较场必须是形状相同的二维数组。")
    return np.max(np.abs(reference - refined), axis=1)


def validate_comparable_outputs(
    reference_times: np.ndarray,
    refined_times: np.ndarray,
    reference_radii: np.ndarray,
    refined_radii: np.ndarray,
) -> None:
    """确认两个结果可以在相同时间和径向网格上逐点比较。"""
    if not np.allclose(reference_times, refined_times):
        raise ValueError("细化结果的时间轴与基准结果不一致。")
    if not np.allclose(reference_radii, refined_radii):
        raise ValueError("细化结果的径向输出网格不一致。")


def snapshot_index(times: np.ndarray, snapshot_time: float) -> int:
    """返回结果时间轴中指定快照的行号。"""
    matches = np.flatnonzero(np.isclose(times, snapshot_time))
    if len(matches) != 1:
        raise ValueError(f"结果中未找到唯一的 {snapshot_time:g} s 快照。")
    return int(matches[0])


def read_boundary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取附件一的时间、烘房温度和等效外界水分变量。"""
    workbook = load_workbook(ATTACHMENT_ONE, data_only=True, read_only=True)
    worksheet = workbook.active
    rows = [
        row for row in worksheet.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("附件一必须包含时间、温度和水分浓度三列。")
    return data[:, 0], data[:, 1], data[:, 2]


def read_problem_one_result() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """读取 result1.xlsx 中的时间、径向输出点、温度场和含水率场。"""
    workbook = load_workbook(RESULT_ONE, data_only=True, read_only=True)
    if len(workbook.worksheets) < 2:
        raise ValueError("result1.xlsx 必须包含温度和含水率两个工作表。")
    temperature_sheet, moisture_sheet = workbook.worksheets[:2]

    def read_sheet(sheet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        rows = [
            row for row in sheet.iter_rows(min_row=2, values_only=True)
            if row[0] is not None
        ]
        times = np.asarray([row[0] for row in rows], dtype=float)
        radii_cm = np.asarray(list(header)[1:], dtype=float)
        field = np.asarray([row[1:] for row in rows], dtype=float)
        return times, radii_cm, field

    times_s, radii_cm, temperature_field = read_sheet(temperature_sheet)
    moisture_times_s, moisture_radii_cm, moisture_field = read_sheet(moisture_sheet)
    validate_comparable_outputs(
        times_s,
        moisture_times_s,
        radii_cm,
        moisture_radii_cm,
    )
    return times_s, radii_cm, temperature_field, moisture_field


@lru_cache(maxsize=1)
def load_plot_data() -> dict[str, object]:
    """读取官方结果并补充精确平均量与空间/时间细化结果。"""
    boundary_times_s, boundary_temperature, boundary_moisture = read_boundary_data()
    workbook_times_s, workbook_radii_cm, workbook_temperature, workbook_moisture = (
        read_problem_one_result()
    )

    reference = problem1_solver.simulate_problem_one()
    reference_radii_cm = reference.output_radii * 100.0
    validate_comparable_outputs(
        reference.times,
        workbook_times_s,
        reference_radii_cm,
        workbook_radii_cm,
    )
    rounding_tolerance = 5.1e-5
    if not np.allclose(
        reference.temperature_field,
        workbook_temperature,
        rtol=0.0,
        atol=rounding_tolerance,
    ):
        raise ValueError("求解器温度场与 result1.xlsx 的四位小数结果不一致。")
    if not np.allclose(
        reference.moisture_field,
        workbook_moisture,
        rtol=0.0,
        atol=rounding_tolerance,
    ):
        raise ValueError("求解器含水率场与 result1.xlsx 的四位小数结果不一致。")

    fine_space = problem1_solver.simulate_problem_one(cell_count=320)
    fine_time = problem1_solver.simulate_problem_one(time_step=0.5)
    for refined in (fine_space, fine_time):
        validate_comparable_outputs(
            reference.times,
            refined.times,
            reference.output_radii,
            refined.output_radii,
        )

    return {
        "boundary_times_s": boundary_times_s,
        "boundary_temperature": boundary_temperature,
        "boundary_moisture": boundary_moisture,
        "times_s": reference.times,
        "radii_cm": reference_radii_cm,
        "temperature_field": reference.temperature_field,
        "moisture_field": reference.moisture_field,
        "average_temperature": reference.average_temperature,
        "average_moisture": reference.average_moisture,
        "diagnostics": reference.diagnostics,
        "fine_space_temperature": fine_space.temperature_field,
        "fine_space_moisture": fine_space.moisture_field,
        "fine_time_temperature": fine_time.temperature_field,
        "fine_time_moisture": fine_time.moisture_field,
    }


def save_publication_figure(
    figure: plt.Figure,
    base_name: str,
    exclude_axes: list[plt.Axes] | None = None,
) -> None:
    """通过面板对齐门禁后仅导出 PNG。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    stem = FIGURE_DIR / base_name
    with TemporaryDirectory(prefix="problem1-figure-qa-") as temporary_dir:
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
    save_kwargs = {"bbox_inches": "tight", "pad_inches": PAD_INCHES, "facecolor": "white"}
    figure.savefig(stem.with_suffix(".png"), dpi=EXPORT_DPI, **save_kwargs)
    plt.close(figure)


def close_without_export(figure: plt.Figure) -> None:
    """关闭测试或预览图，不触发文件导出。"""
    plt.close(figure)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.set_axisbelow(True)
    axis.tick_params(length=3, width=0.7)
    for spine in (axis.spines["left"], axis.spines["bottom"]):
        spine.set_color("#51474D")
        spine.set_linewidth(0.75)


def plot_boundary_conditions(data: dict[str, object]) -> plt.Figure:
    """绘制实测烘房边界及问题一求解时间窗口。"""
    boundary_times_h = np.asarray(data["boundary_times_s"], dtype=float) / 3600.0
    boundary_temperature = np.asarray(data["boundary_temperature"], dtype=float)
    boundary_moisture = np.asarray(data["boundary_moisture"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    series = (
        (axes[0], boundary_temperature, "T∞ / °C", "温度边界", COLOR_CENTER, "a"),
        (
            axes[1],
            boundary_moisture,
            "C∞ / kg/kg",
            "水分边界",
            COLOR_SURFACE,
            "b",
        ),
    )
    for axis, values, ylabel, title, color, label in series:
        axis.plot(boundary_times_h, values, color=color, linewidth=1.7)
        axis.set_title(title, loc="left", pad=7, fontsize=9.2, fontweight="medium")
        axis.set_xlabel("时间 / h")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 0.5)
        _style_axis(axis)
    figure.suptitle("实测边界条件", x=0.5, y=0.96, fontsize=12, fontweight="bold")
    figure.subplots_adjust(wspace=0.30, bottom=0.21, top=0.73, left=0.09, right=0.98)
    return figure


def _plot_position_trajectories(
    axis: plt.Axes,
    times_h: np.ndarray,
    field: np.ndarray,
    average: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    for values, label, color, linestyle in (
        (field[:, 0], "中心", COLOR_CENTER, "-"),
        (average, "体积平均", COLOR_AVERAGE, "--"),
        (field[:, -1], "表面", COLOR_SURFACE, "-"),
    ):
        axis.plot(
            times_h,
            values,
            color=color,
            linewidth=1.55 if label != "体积平均" else 1.25,
            linestyle=linestyle,
            label=label,
        )
    axis.set_title(title, loc="left", pad=7, fontsize=9.2, fontweight="medium")
    axis.set_xlabel("时间 / h")
    axis.set_ylabel(ylabel)
    axis.set_xlim(times_h[0], times_h[-1] + 0.045)
    _style_axis(axis)


def plot_main_response(data: dict[str, object]) -> plt.Figure:
    """绘制中心、体积平均和表面处的温湿响应。"""
    times_h = np.asarray(data["times_s"], dtype=float) / 3600.0
    temperature_field = np.asarray(data["temperature_field"], dtype=float)
    moisture_field = np.asarray(data["moisture_field"], dtype=float)
    average_temperature = np.asarray(data["average_temperature"], dtype=float)
    average_moisture = np.asarray(data["average_moisture"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    _plot_position_trajectories(
        axes[0], times_h, temperature_field, average_temperature, "温度 / °C", "温度"
    )
    _plot_position_trajectories(
        axes[1],
        times_h,
        moisture_field,
        average_moisture,
        "干基含水率 / kg/kg",
        "含水率",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("药材内部温湿响应", x=0.5, y=0.97, fontsize=12, fontweight="bold")
    figure.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        handlelength=1.8,
        columnspacing=1.5,
    )
    figure.subplots_adjust(wspace=0.31, bottom=0.21, top=0.68, left=0.09, right=0.98)
    return figure


def plot_field_evolution(data: dict[str, object]) -> plt.Figure:
    """绘制加入 t=0 初始状态的温度和含水率时空场。"""
    times_s = np.asarray(data["times_s"], dtype=float)
    radii_cm = np.asarray(data["radii_cm"], dtype=float)
    temperature_times, temperature_field = prepend_initial_state(
        times_s,
        np.asarray(data["temperature_field"], dtype=float),
        problem1_solver.INITIAL_TEMPERATURE,
    )
    moisture_times, moisture_field = prepend_initial_state(
        times_s,
        np.asarray(data["moisture_field"], dtype=float),
        problem1_solver.INITIAL_MOISTURE,
    )
    if not np.allclose(temperature_times, moisture_times):
        raise ValueError("温度场和含水率场的时空网格不一致。")
    times_h = temperature_times / 3600.0
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, sharey=True)
    colorbars = []
    panels = (
        (axes[0], temperature_field, "magma", "温度场", "°C", "a"),
        (
            axes[1],
            moisture_field,
            "viridis",
            "含水率场",
            "kg/kg",
            "b",
        ),
    )
    for axis, field, cmap, title, colorbar_label, label in panels:
        mesh = axis.pcolormesh(
            radii_cm,
            times_h,
            field,
            shading="auto",
            cmap=cmap,
            vmin=float(np.min(field)),
            vmax=float(np.max(field)),
        )
        colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label, rotation=270, labelpad=13)
        colorbars.append(colorbar.ax)
        axis.set_title(title, loc="left", pad=7, fontsize=9.2, fontweight="medium")
        axis.set_xlabel("半径 r / cm")
        axis.set_ylabel("时间 / h")
        axis.set_xlim(radii_cm[0], radii_cm[-1])
        axis.set_ylim(times_h[0], times_h[-1])
        axis.grid(False)
    figure.suptitle("热湿场时空演化", x=0.5, y=0.97, fontsize=12, fontweight="bold")
    figure._alignment_exclude_axes = colorbars
    # colorbar 会把两个热图拆为独立子网格，因此显式声明它们属于同一行。
    figure._alignment_row_groups = [["a", "b"]]
    figure.subplots_adjust(wspace=0.28, bottom=0.17, top=0.75, left=0.09, right=0.92)
    return figure


def plot_radial_profiles(data: dict[str, object]) -> plt.Figure:
    """绘制四个代表时刻的温度和含水率径向剖面。"""
    times_s = np.asarray(data["times_s"], dtype=float)
    radii_cm = np.asarray(data["radii_cm"], dtype=float)
    temperature_field = np.asarray(data["temperature_field"], dtype=float)
    moisture_field = np.asarray(data["moisture_field"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharex=True)
    lines = []
    panels = (
        (axes[0], temperature_field, "温度 / °C", "温度", "a"),
        (
            axes[1],
            moisture_field,
            "干基含水率 / kg/kg",
            "含水率",
            "b",
        ),
    )
    for axis, field, ylabel, title, label in panels:
        for color, snapshot_time in zip(PROFILE_COLORS, SNAPSHOT_TIMES):
            (line,) = axis.plot(
                radii_cm,
                field[snapshot_index(times_s, snapshot_time)],
                color=color,
                linewidth=1.65,
                label=f"{snapshot_time:g} s",
            )
            if axis is axes[0]:
                lines.append(line)
        axis.set_title(title, loc="left", pad=7, fontsize=9.2, fontweight="medium")
        axis.set_xlabel("半径 r / cm")
        axis.set_ylabel(ylabel)
        axis.set_xlim(radii_cm[0], radii_cm[-1])
        _style_axis(axis)
    figure.suptitle("关键时刻径向剖面", x=0.5, y=0.97, fontsize=12, fontweight="bold")
    figure.legend(
        handles=lines,
        labels=[f"{value:g} s" for value in SNAPSHOT_TIMES],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.87),
        handlelength=1.9,
        columnspacing=1.3,
    )
    figure.subplots_adjust(wspace=0.30, bottom=0.17, top=0.70, left=0.10, right=0.98)
    return figure


def build_validation_errors(data: dict[str, object]) -> dict[str, np.ndarray]:
    """提取空间网格和时间步长细化的逐时刻最大径向误差。"""
    reference_temperature = np.asarray(data["temperature_field"], dtype=float)
    reference_moisture = np.asarray(data["moisture_field"], dtype=float)
    return {
        "grid_temperature": max_radial_error(
            reference_temperature,
            np.asarray(data["fine_space_temperature"], dtype=float),
        ),
        "grid_moisture": max_radial_error(
            reference_moisture,
            np.asarray(data["fine_space_moisture"], dtype=float),
        ),
        "time_temperature": max_radial_error(
            reference_temperature,
            np.asarray(data["fine_time_temperature"], dtype=float),
        ),
        "time_moisture": max_radial_error(
            reference_moisture,
            np.asarray(data["fine_time_moisture"], dtype=float),
        ),
    }


def plot_numerical_validation(data: dict[str, object]) -> plt.Figure:
    """绘制空间/时间细化误差和基准求解诊断。"""
    times_h = np.asarray(data["times_s"], dtype=float) / 3600.0
    errors = build_validation_errors(data)
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.55), sharex=True)
    panels = (
        (
            axes[0, 0],
            errors["grid_temperature"],
            "空间加密｜温度",
            "最大径向误差 / °C",
            "N = 160 vs N = 320",
            "a",
        ),
        (
            axes[0, 1],
            errors["grid_moisture"],
            "空间加密｜含水率",
            "最大径向误差 / kg/kg",
            "N = 160 vs N = 320",
            "b",
        ),
        (
            axes[1, 0],
            errors["time_temperature"],
            "时间加密｜温度",
            "最大径向误差 / °C",
            "Δt = 1.0 s vs 0.5 s",
            "c",
        ),
        (
            axes[1, 1],
            errors["time_moisture"],
            "时间加密｜含水率",
            "最大径向误差 / kg/kg",
            "Δt = 1.0 s vs 0.5 s",
            "d",
        ),
    )
    for axis, values, title, ylabel, comparison, label in panels:
        axis.semilogy(times_h, values, color=COLOR_TEAL, linewidth=1.45)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.0e}"))
        axis.set_title(title, loc="left", pad=7, fontsize=9.2, fontweight="medium")
        axis.set_xlabel("时间 / h")
        axis.set_ylabel(ylabel)
        axis.set_xlim(times_h[0], times_h[-1])
        axis.text(
            0.02,
            0.96,
            comparison.replace(" vs ", " → "),
            transform=axis.transAxes,
            fontsize=6.4,
            ha="left",
            va="top",
            color=COLOR_TEAL,
        )
        _style_axis(axis)
    figure.suptitle("数值收敛验证", x=0.5, y=0.97, fontsize=12, fontweight="bold")
    figure.subplots_adjust(
        wspace=0.34,
        hspace=0.38,
        bottom=0.14,
        top=0.84,
        left=0.10,
        right=0.98,
    )
    return figure


LEGACY_PROBLEM_ONE_OUTPUTS = (
    "图1_附件一烘房温度曲线.png",
    "图2_附件一烘房水分浓度曲线.png",
    "图3_附件一边界采样间隔.png",
    "图4_温度径向剖面.png",
    "图5_水分径向剖面.png",
    "图6_中心与表面轨迹.png",
    "图7_温度场时空分布.png",
    "图8_水分场时空分布.png",
    "图9_空间非均匀性.png",
)


def remove_legacy_problem_one_outputs() -> None:
    """只清理旧问题一图集的精确文件名。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for basename in LEGACY_PROBLEM_ONE_OUTPUTS:
        target = FIGURE_DIR / basename
        if target.parent != FIGURE_DIR:
            raise ValueError(f"拒绝清理图集目录之外的路径：{target}")
        if target.exists():
            target.unlink()


def export_all(data: dict[str, object]) -> None:
    """生成五张新图，确认导出完整后再清理旧问题一 PNG。"""
    builders = (
        (plot_boundary_conditions, NEW_FIGURE_BASES[0]),
        (plot_main_response, NEW_FIGURE_BASES[1]),
        (plot_field_evolution, NEW_FIGURE_BASES[2]),
        (plot_radial_profiles, NEW_FIGURE_BASES[3]),
        (plot_numerical_validation, NEW_FIGURE_BASES[4]),
    )
    for builder, base_name in builders:
        figure = builder(data)
        save_publication_figure(
            figure,
            base_name,
            exclude_axes=list(getattr(figure, "_alignment_exclude_axes", [])),
        )

    missing_outputs = [
        FIGURE_DIR / f"{base_name}.{suffix}"
        for base_name in NEW_FIGURE_BASES
        for suffix in FIGURE_FORMATS
        if not (FIGURE_DIR / f"{base_name}.{suffix}").exists()
    ]
    if missing_outputs:
        raise FileNotFoundError(f"新图集导出不完整：{missing_outputs}")
    remove_legacy_problem_one_outputs()
    print(
        f"问题一绘图完成：生成 {len(NEW_FIGURE_BASES)} 张图，"
        f"每张包含 {len(FIGURE_FORMATS)} 种格式，输出目录：{FIGURE_DIR}"
    )


def main() -> None:
    export_all(load_plot_data())


if __name__ == "__main__":
    main()
