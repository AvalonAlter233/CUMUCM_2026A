"""问题二论文配图：模型影响、物性耦合、热湿响应与数值验证。

主图数据来自 result1.xlsx 与 result2.xlsx；离散误差采用问题2_求解.py
--validate 对应的全 3 h 验证结果。脚本只导出 600 dpi PNG。
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from openpyxl import load_workbook

import 问题2_求解 as problem2_solver


NATURE_FIGURE_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(NATURE_FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NATURE_FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_ONE = PROJECT_ROOT / "附件" / "附件3" / "result1.xlsx"
RESULT_TWO = PROJECT_ROOT / "附件" / "附件3" / "result2.xlsx"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题二"

CJK_FONT_CANDIDATES = (
    "STSong", "SimSun", "Source Han Serif SC", "Noto Serif CJK SC",
)
INSTALLED_FONTS = {font.name for font in font_manager.fontManager.ttflist}
AVAILABLE_CJK_FONTS = [name for name in CJK_FONT_CANDIDATES if name in INSTALLED_FONTS]
if not AVAILABLE_CJK_FONTS:
    print("提示：本机未检测到中文字体，图中中文可能显示为方框。")
HEADING_FONT = "STZhongsong" if "STZhongsong" in INSTALLED_FONTS else AVAILABLE_CJK_FONTS[0]

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [*AVAILABLE_CJK_FONTS, "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.labelweight": "bold",
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 120,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

EXPORT_DPI = 600
PAD_INCHES = 0.06
FIGURE_FORMATS = ("png",)

REFERENCE_PALETTE = ("#44757A", "#452A3D", "#D44C3C", "#EED5B7")
COLOR_TEAL, COLOR_PLUM, COLOR_CORAL, COLOR_SAND = REFERENCE_PALETTE
COLOR_CENTER = COLOR_TEAL
COLOR_AVERAGE = COLOR_PLUM
COLOR_SURFACE = COLOR_CORAL

NEW_FIGURE_BASES = (
    "图1_常物性与变物性模型对比",
    "图2_物性参数耦合演化",
    "图3_中心平均表面响应",
    "图4_热湿场时空演化",
    "图5_数值收敛验证",
)

# 问题2_求解.py --validate 的全 3 h 运行结果；每项依次为
# “全时段最大绝对差、3 h 终点最大绝对差”。
VALIDATION_METRICS = {
    "grid_temperature": (1.16e-4, 9.58e-6),
    "grid_moisture": (1.67e-2, 1.71e-5),
    "time_temperature": (8.76e-4, 3.53e-5),
    "time_moisture": (7.54e-4, 5.58e-6),
}


def prepend_initial_state(
    times: np.ndarray, field: np.ndarray, initial_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """把已知的 t=0 均匀初始场加入结果序列。"""
    if field.ndim != 2 or field.shape[0] != len(times):
        raise ValueError("时间轴与场数组的第一维不一致。")
    initial_row = np.full((1, field.shape[1]), float(initial_value))
    return np.concatenate(([0.0], times)), np.vstack((initial_row, field))


def radial_area_average(radii_cm: np.ndarray, field: np.ndarray) -> np.ndarray:
    """按圆柱截面面积权重计算径向场平均值。"""
    radii_cm = np.asarray(radii_cm, dtype=float)
    field = np.asarray(field, dtype=float)
    if radii_cm.ndim != 1 or len(radii_cm) < 2:
        raise ValueError("径向坐标必须是一维且至少包含两个点。")
    if not np.all(np.diff(radii_cm) > 0.0):
        raise ValueError("径向坐标必须严格递增。")
    if field.ndim != 2 or field.shape[1] != len(radii_cm):
        raise ValueError("场数组列数必须与径向坐标长度一致。")
    weighted = field * radii_cm[np.newaxis, :]
    integral = np.trapezoid(weighted, radii_cm, axis=1)
    return 2.0 * integral / radii_cm[-1] ** 2


def read_field_workbook(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取题目结果工作簿中的温度场和含水率场。"""
    workbook = load_workbook(path, data_only=True, read_only=True)
    if len(workbook.worksheets) < 2:
        raise ValueError(f"{path.name} 必须包含温度和含水率两个工作表。")

    def read_sheet(sheet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        rows = [
            row
            for row in sheet.iter_rows(min_row=2, values_only=True)
            if row[0] is not None
        ]
        times_s = np.asarray([row[0] for row in rows], dtype=float)
        radii_cm = np.asarray(list(header)[1:], dtype=float)
        field = np.asarray([row[1:] for row in rows], dtype=float)
        return times_s, radii_cm, field

    times_s, radii_cm, temperature = read_sheet(workbook.worksheets[0])
    moisture_times_s, moisture_radii_cm, moisture = read_sheet(workbook.worksheets[1])
    if not np.allclose(times_s, moisture_times_s):
        raise ValueError(f"{path.name} 两个工作表的时间轴不一致。")
    if not np.allclose(radii_cm, moisture_radii_cm):
        raise ValueError(f"{path.name} 两个工作表的径向坐标不一致。")
    return times_s, radii_cm, temperature, moisture


def load_plot_data() -> dict[str, np.ndarray]:
    """读取问题一、问题二结果并构造论文绘图所需量。"""
    q1_times_s, q1_radii_cm, q1_temperature, q1_moisture = read_field_workbook(RESULT_ONE)
    times_s, radii_cm, temperature, moisture = read_field_workbook(RESULT_TWO)
    if not np.all(np.diff(times_s) > 0.0):
        raise ValueError("问题二结果时间轴必须严格递增。")
    if not np.allclose(q1_radii_cm, radii_cm):
        raise ValueError("问题一、问题二的径向输出网格不一致。")
    q2_comparison_mask = times_s <= q1_times_s[-1]
    if not np.allclose(times_s[q2_comparison_mask], q1_times_s):
        raise ValueError("问题一、问题二在共同时间窗内的输出时刻不一致。")

    return {
        "times_s": times_s,
        "radii_cm": radii_cm,
        "temperature_field": temperature,
        "moisture_field": moisture,
        "average_temperature": radial_area_average(radii_cm, temperature),
        "average_moisture": radial_area_average(radii_cm, moisture),
        "q1_times_s": q1_times_s,
        "q1_temperature_field": q1_temperature,
        "q1_moisture_field": q1_moisture,
    }


def save_publication_figure(
    figure: plt.Figure,
    base_name: str,
    exclude_axes: list[plt.Axes] | None = None,
) -> None:
    """通过面板对齐检查后，仅导出 600 dpi PNG。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    with TemporaryDirectory(prefix="problem2-figure-qa-") as temporary_dir:
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
        FIGURE_DIR / f"{base_name}.png",
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pad_inches=PAD_INCHES,
        facecolor="white",
    )
    plt.close(figure)


def close_without_export(figure: plt.Figure) -> None:
    """关闭测试或预览图，不生成文件。"""
    plt.close(figure)


def _style_axis(axis: plt.Axes, *, show_grid: bool = True) -> None:
    axis.set_axisbelow(True)
    if show_grid:
        axis.grid(
            True,
            which="major",
            color="#AAA5A8",
            linestyle="--",
            linewidth=0.55,
            alpha=0.70,
        )
    else:
        axis.grid(False)
    axis.tick_params(length=3, width=0.7)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        if not any("\u4e00" <= character <= "\u9fff" for character in label.get_text()):
            label.set_fontfamily("Times New Roman")
    for text in (axis.title, axis.xaxis.label, axis.yaxis.label):
        text.set_fontfamily(HEADING_FONT)
        text.set_fontweight("bold")
    for spine in (axis.spines["left"], axis.spines["bottom"]):
        spine.set_color("#51474D")
        spine.set_linewidth(0.75)


def _set_top_title(figure: plt.Figure, title: str, y: float = 0.97) -> None:
    text = figure.suptitle(title, x=0.5, y=y, fontsize=12, fontweight="bold")
    text.set_fontfamily(HEADING_FONT)


def _plot_center_surface(
    axis: plt.Axes,
    times_h: np.ndarray,
    field: np.ndarray,
    linestyle: str,
    model_name: str,
) -> list[plt.Line2D]:
    lines = []
    for values, position, color in (
        (field[:, 0], "中心", COLOR_CENTER),
        (field[:, -1], "表面", COLOR_SURFACE),
    ):
        (line,) = axis.plot(
            times_h,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=1.7 if linestyle == "-" else 1.3,
            alpha=1.0 if linestyle == "-" else 0.75,
            label=f"{position}·{model_name}",
        )
        lines.append(line)
    return lines


def plot_model_comparison(data: dict[str, np.ndarray]) -> plt.Figure:
    """比较共同 0–0.5 h 时间窗内常物性与变物性模型。"""
    q1_times_s = np.asarray(data["q1_times_s"], dtype=float)
    q2_times_s = np.asarray(data["times_s"], dtype=float)
    if q1_times_s[-1] > q2_times_s[-1]:
        raise ValueError("问题二结果没有覆盖问题一的比较时间窗。")
    q2_temperature_source = np.asarray(data["temperature_field"], dtype=float)
    q2_moisture_source = np.asarray(data["moisture_field"], dtype=float)
    q2_temperature_common = np.column_stack(
        [np.interp(q1_times_s, q2_times_s, q2_temperature_source[:, column])
         for column in range(q2_temperature_source.shape[1])]
    )
    q2_moisture_common = np.column_stack(
        [np.interp(q1_times_s, q2_times_s, q2_moisture_source[:, column])
         for column in range(q2_moisture_source.shape[1])]
    )
    q1_t, q1_temperature = prepend_initial_state(
        q1_times_s,
        np.asarray(data["q1_temperature_field"], dtype=float),
        problem2_solver.INITIAL_TEMPERATURE,
    )
    _, q1_moisture = prepend_initial_state(
        q1_times_s,
        np.asarray(data["q1_moisture_field"], dtype=float),
        problem2_solver.INITIAL_MOISTURE,
    )
    q2_t, q2_temperature = prepend_initial_state(
        q1_times_s,
        q2_temperature_common,
        problem2_solver.INITIAL_TEMPERATURE,
    )
    _, q2_moisture = prepend_initial_state(
        q1_times_s,
        q2_moisture_common,
        problem2_solver.INITIAL_MOISTURE,
    )
    if not np.allclose(q1_t, q2_t):
        raise ValueError("问题一、问题二的共同时间窗不一致。")

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    panels = (
        (axes[0], q1_temperature, q2_temperature, "温度 / °C", "温度响应"),
        (axes[1], q1_moisture, q2_moisture, "干基含水率 / kg/kg", "水分响应"),
    )
    legend_lines = []
    for axis, constant_field, variable_field, ylabel, title in panels:
        constant_lines = _plot_center_surface(axis, q1_t / 3600.0, constant_field, "--", "常物性")
        variable_lines = _plot_center_surface(axis, q2_t / 3600.0, variable_field, "-", "变物性")
        if axis is axes[0]:
            legend_lines = [variable_lines[0], constant_lines[0], variable_lines[1], constant_lines[1]]
        axis.set_title(title, pad=6, fontweight="bold", fontfamily=HEADING_FONT)
        axis.set_xlabel("时间 / h")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 0.5)
        _style_axis(axis)
    _set_top_title(figure, "常物性与变物性模型对比")
    figure.legend(
        handles=legend_lines,
        labels=[line.get_label() for line in legend_lines],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        handlelength=2.0,
        columnspacing=1.15,
    )
    figure.subplots_adjust(wspace=0.30, bottom=0.20, top=0.67, left=0.09, right=0.98)
    figure._alignment_row_groups = [["a", "b"]]
    return figure


def plot_property_coupling(data: dict[str, np.ndarray]) -> plt.Figure:
    """展示含水率和温度变化对关键物性的反馈。"""
    times_s, temperature = prepend_initial_state(
        np.asarray(data["times_s"], dtype=float),
        np.asarray(data["temperature_field"], dtype=float),
        problem2_solver.INITIAL_TEMPERATURE,
    )
    _, moisture = prepend_initial_state(
        np.asarray(data["times_s"], dtype=float),
        np.asarray(data["moisture_field"], dtype=float),
        problem2_solver.INITIAL_MOISTURE,
    )
    volumetric_heat_capacity = (
        problem2_solver.density(moisture) * problem2_solver.heat_capacity(moisture)
    ) / 1.0e6
    diffusivity = problem2_solver.moisture_diffusivity(moisture, temperature) * 1.0e9

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharex=True)
    lines = []
    panels = (
        (axes[0], volumetric_heat_capacity, "体积热容 / 10^6 J/(m^3·K)", "热储存能力"),
        (axes[1], diffusivity, "扩散系数 D / 10^-9 m^2/s", "水分扩散能力"),
    )
    for axis, values, ylabel, title in panels:
        for column, label, color in ((0, "中心", COLOR_CENTER), (-1, "表面", COLOR_SURFACE)):
            (line,) = axis.plot(
                times_s / 3600.0,
                values[:, column],
                color=color,
                linewidth=1.65,
                label=label,
            )
            if axis is axes[0]:
                lines.append(line)
        axis.set_title(title, pad=6, fontweight="bold", fontfamily=HEADING_FONT)
        axis.set_xlabel("时间 / h")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 3.0)
        _style_axis(axis)
    _set_top_title(figure, "物性参数耦合演化")
    figure.legend(
        handles=lines,
        labels=[line.get_label() for line in lines],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        columnspacing=1.5,
    )
    figure.subplots_adjust(wspace=0.32, bottom=0.20, top=0.67, left=0.11, right=0.98)
    figure._alignment_row_groups = [["a", "b"]]
    return figure


def plot_main_response(data: dict[str, np.ndarray]) -> plt.Figure:
    """绘制中心、截面平均和表面的温湿响应。"""
    times_s, temperature = prepend_initial_state(
        np.asarray(data["times_s"], dtype=float),
        np.asarray(data["temperature_field"], dtype=float),
        problem2_solver.INITIAL_TEMPERATURE,
    )
    _, moisture = prepend_initial_state(
        np.asarray(data["times_s"], dtype=float),
        np.asarray(data["moisture_field"], dtype=float),
        problem2_solver.INITIAL_MOISTURE,
    )
    average_temperature = np.concatenate(([problem2_solver.INITIAL_TEMPERATURE], np.asarray(data["average_temperature"])))
    average_moisture = np.concatenate(([problem2_solver.INITIAL_MOISTURE], np.asarray(data["average_moisture"])))

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    lines = []
    panels = (
        (axes[0], temperature, average_temperature, "温度 / °C", "温度"),
        (axes[1], moisture, average_moisture, "干基含水率 / kg/kg", "含水率"),
    )
    for axis, field, average, ylabel, title in panels:
        for values, label, color, linestyle in (
            (field[:, 0], "中心", COLOR_CENTER, "-"),
            (average, "体积平均", COLOR_AVERAGE, "--"),
            (field[:, -1], "表面", COLOR_SURFACE, "-"),
        ):
            (line,) = axis.plot(
                times_s / 3600.0,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=1.65 if linestyle == "-" else 1.3,
                label=label,
            )
            if axis is axes[0]:
                lines.append(line)
        axis.set_title(title, pad=6, fontweight="bold", fontfamily=HEADING_FONT)
        axis.set_xlabel("时间 / h")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 3.0)
        _style_axis(axis)
    _set_top_title(figure, "药材内部温湿响应")
    figure.legend(
        handles=lines,
        labels=[line.get_label() for line in lines],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        columnspacing=1.4,
    )
    figure.subplots_adjust(wspace=0.30, bottom=0.20, top=0.67, left=0.09, right=0.98)
    figure._alignment_row_groups = [["a", "b"]]
    return figure


def plot_field_evolution(data: dict[str, np.ndarray]) -> plt.Figure:
    """绘制包含 t=0 初始状态的温度场与含水率场。"""
    times_s, temperature = prepend_initial_state(
        np.asarray(data["times_s"], dtype=float),
        np.asarray(data["temperature_field"], dtype=float),
        problem2_solver.INITIAL_TEMPERATURE,
    )
    _, moisture = prepend_initial_state(
        np.asarray(data["times_s"], dtype=float),
        np.asarray(data["moisture_field"], dtype=float),
        problem2_solver.INITIAL_MOISTURE,
    )
    radii_cm = np.asarray(data["radii_cm"], dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), sharex=True, sharey=True)
    colorbar_axes = []
    panels = (
        (axes[0], temperature, "magma", "温度场", "°C"),
        (axes[1], moisture, "viridis", "含水率场", "kg/kg"),
    )
    for axis, field, cmap, title, colorbar_label in panels:
        mesh = axis.pcolormesh(
            radii_cm,
            times_s / 3600.0,
            field,
            shading="auto",
            cmap=cmap,
            vmin=float(np.min(field)),
            vmax=float(np.max(field)),
        )
        colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label, rotation=270, labelpad=12)
        colorbar.ax.yaxis.label.set_fontfamily(HEADING_FONT)
        colorbar.ax.yaxis.label.set_fontweight("bold")
        for tick in colorbar.ax.get_yticklabels():
            tick.set_fontfamily("Times New Roman")
        colorbar.ax.grid(False)
        colorbar_axes.append(colorbar.ax)
        axis.set_title(title, pad=6, fontweight="bold", fontfamily=HEADING_FONT)
        axis.set_xlabel("半径 r / cm")
        axis.set_ylabel("时间 / h")
        axis.set_xlim(radii_cm[0], radii_cm[-1])
        axis.set_ylim(0.0, 3.0)
        _style_axis(axis, show_grid=False)
    _set_top_title(figure, "热湿场时空演化")
    figure._alignment_exclude_axes = colorbar_axes
    figure._alignment_row_groups = [["a", "b"]]
    figure.subplots_adjust(wspace=0.28, bottom=0.17, top=0.78, left=0.09, right=0.92)
    return figure


def plot_numerical_validation(data: dict[str, np.ndarray]) -> plt.Figure:
    """汇总全时段与 3 h 终点的空间、时间离散误差。"""
    del data
    categories = np.arange(2)
    category_labels = ("全时段最大", "3 h 终点")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    panels = (
        (axes[0], VALIDATION_METRICS["grid_temperature"], VALIDATION_METRICS["time_temperature"], "最大绝对差 / °C", "温度场"),
        (axes[1], VALIDATION_METRICS["grid_moisture"], VALIDATION_METRICS["time_moisture"], "最大绝对差 / kg/kg", "含水率场"),
    )
    legend_lines = []
    for axis, grid_values, time_values, ylabel, title in panels:
        for values, label, color, marker in (
            (grid_values, "空间网格加密", COLOR_TEAL, "o"),
            (time_values, "时间步长减半", COLOR_PLUM, "s"),
        ):
            (line,) = axis.plot(
                categories,
                values,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.5,
                label=label,
            )
            if axis is axes[0]:
                legend_lines.append(line)
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.0e}"))
        axis.set_xticks(categories, category_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=6, fontweight="bold", fontfamily=HEADING_FONT)
        axis.set_xlim(-0.15, 1.15)
        _style_axis(axis)
    _set_top_title(figure, "空间与时间离散检验")
    figure.legend(
        handles=legend_lines,
        labels=[line.get_label() for line in legend_lines],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        columnspacing=1.5,
    )
    figure.subplots_adjust(wspace=0.31, bottom=0.19, top=0.67, left=0.11, right=0.98)
    figure._alignment_row_groups = [["a", "b"]]
    return figure


LEGACY_PROBLEM_TWO_OUTPUTS = (
    "图1_烘房温度与水分边界.png",
    "图2_边界采样间隔.png",
    "图3_边界变化率.png",
    "图4_温度径向剖面.png",
    "图5_含水率径向剖面.png",
    "图6_中心与表面轨迹.png",
    "图7_温度场时空分布.png",
    "图8_含水率场时空分布.png",
    "图9_空间非均匀性.png",
)


def remove_legacy_problem_two_outputs() -> None:
    """新图完整生成后，只清理旧问题二图集的精确文件名。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for basename in LEGACY_PROBLEM_TWO_OUTPUTS:
        target = FIGURE_DIR / basename
        if target.parent != FIGURE_DIR:
            raise ValueError(f"拒绝清理图集目录之外的路径：{target}")
        if target.exists():
            target.unlink()


def export_all(data: dict[str, np.ndarray]) -> None:
    """生成五张论文图，确认齐全后清理旧图。"""
    builders = (
        plot_model_comparison,
        plot_property_coupling,
        plot_main_response,
        plot_field_evolution,
        plot_numerical_validation,
    )
    for builder, base_name in zip(builders, NEW_FIGURE_BASES):
        figure = builder(data)
        save_publication_figure(
            figure,
            base_name,
            exclude_axes=list(getattr(figure, "_alignment_exclude_axes", [])),
        )

    missing = [
        FIGURE_DIR / f"{base_name}.png"
        for base_name in NEW_FIGURE_BASES
        if not (FIGURE_DIR / f"{base_name}.png").exists()
    ]
    if missing:
        raise RuntimeError(f"新问题二图集导出不完整：{missing}")
    remove_legacy_problem_two_outputs()


def main() -> None:
    export_all(load_plot_data())
    print(f"问题二绘图完成，共生成 {len(NEW_FIGURE_BASES)} 张 PNG：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
