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

if (Path(__file__).parent / "问题3_求解_改.py").is_file():
    import 问题3_求解_改 as problem3_solver
else:
    import 问题3_求解 as problem3_solver


nature_figure_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(nature_figure_scripts) not in sys.path:
    sys.path.insert(0, str(nature_figure_scripts))
from audit_panel_alignment import require_matplotlib_panel_alignment


project_root = Path(__file__).resolve().parent
boundary_file = project_root / "附件" / "附件1.xlsx"
result_file = project_root / "附件" / "附件3" / "result3.xlsx"
diagnostics_file = project_root / "附件" / "附件3" / "result3_diagnostics.json"
figure_dir = project_root / "figures" / "问题三"

default_critical_moisture = 0.15
initial_moisture = 2.55
export_dpi = 600
pad_inches = 0.06
figure_formats = ("png",)

cjk_font_candidates = (
    "STSong", "SimSun", "Source Han Serif SC", "Noto Serif CJK SC",
)
installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
available_cjk_fonts = [name for name in cjk_font_candidates if name in installed_fonts]
heading_font = "STZhongsong" if "STZhongsong" in installed_fonts else available_cjk_fonts[0]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [*available_cjk_fonts, "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 9.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
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

reference_palette = ("#44757A", "#452A3D", "#D44C3C", "#EED5B7")
color_teal, color_plum, color_coral, color_sand = reference_palette
color_dark = "#51474D"

new_figure_bases = (
    "图1_长期边界平台依据",
    "图2_全域达标时间判定",
    "图3_含水率时空演化",
    "图4_后期拖尾机制",
    "图5_边界与数值稳健性",
)


def read_boundary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 先从附件里把环境数据读出来
    # 时间于此换成小时，接下来画图时就一直拿换算后的时间来用。
    book = load_workbook(boundary_file, data_only=True, read_only=True)
    rows = [row for row in book.active.iter_rows(min_row=2, values_only=True) if row[0] is not None]
    data = np.asarray(rows, dtype=float)
    return data[:, 0] / 3600.0, data[:, 1], data[:, 2]


def read_results() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 整张结果表先读进来，表头和下面的数据分开取。
    # 位置从第一行取，时间和含水率从后面的行取，一起返回。
    book = load_workbook(result_file, data_only=True, read_only=True)
    rows = list(book.active.iter_rows(values_only=True))
    radii_cm = np.asarray(rows[0][1:], dtype=float)
    times_h = np.asarray([row[0] for row in rows[1:]], dtype=float) / 3600.0
    water = np.asarray([row[1:] for row in rows[1:]], dtype=float)
    return times_h, radii_cm, water


def find_threshold_times(
    times_h: np.ndarray,
    maximum_moisture: np.ndarray,
    threshold: float = default_critical_moisture,
) -> tuple[float, float]:
    # 先找第一次低于阈值的那条记录，再将其前一者亦取（）。
    # 两条记录都在后，就在它们之间插一下，再把估计的时间和输出时刻返回。
    # 先有前后这两条记录，再去算夹在中间的时间
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
    # 结果表里的数要用，诊断文件里的一些时间记录亦是如此
    # 这里先把它们放到同一份数据里，后面画到哪一张，就从里面挑出那张需要的几项
    env_hours, air_temp, air_water = read_boundary()
    out_times, radii_cm, out_water = read_results()
    with diagnostics_file.open("r", encoding="utf-8") as stream:
        stats = json.load(stream)
    times_h = np.concatenate(([0.0], out_times))
    water = np.vstack((np.full((1, out_water.shape[1]), initial_moisture), out_water))
    baseline = stats["baseline"]
    return {
        "boundary_times_h": env_hours,
        "room_temperature": air_temp,
        "room_moisture": air_water,
        "times_h": times_h,
        "radii_cm": radii_cm,
        "moisture": water,
        "center_moisture": water[:, 0],
        "surface_moisture": water[:, -1],
        "maximum_moisture": water.max(axis=1),
        "continuous_threshold_time_h": float(baseline["continuous_threshold_time_h"]),
        "discrete_threshold_time_h": float(baseline["discrete_threshold_time_h"]),
        "diagnostics": stats,
    }


def phase_rates(data: dict[str, object]) -> tuple[float, float]:
    # 两个阶段的平均变化速度先分别算出来
    # 等后面要把它们放到图里比较时，直接把这里得到的两个结果拿过去就行。
    # 这两个数先各算各的，返回以后再放到同一张图上一起看。
    times = np.asarray(data["times_h"], dtype=float)
    center = np.asarray(data["center_moisture"], dtype=float)
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("结果时间轴必须严格递增。")
    target = float(data["continuous_threshold_time_h"])
    c12 = float(center[int(np.argmin(np.abs(times - 12.0)))])
    c36 = float(center[int(np.argmin(np.abs(times - 36.0)))])
    return (initial_moisture - c12) / 12.0, (c36 - default_critical_moisture) / (target - 36.0)


def style_axis(axis: plt.Axes, *, show_grid: bool = True) -> None:
    # 这几行主要收拾坐标轴周围的小地方，画完曲线以后接着用
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
        text.set_fontfamily(heading_font)
        text.set_fontweight("bold")
    for name in ("left", "bottom"):
        axis.spines[name].set_color(color_dark)
        axis.spines[name].set_linewidth(0.75)


def set_title(figure: plt.Figure, title: str, y: float = 0.97) -> None:
    text = figure.suptitle(title, x=0.5, y=y, fontsize=12, fontweight="bold")
    text.set_fontfamily(heading_font)


def close_figure(figure: plt.Figure) -> None:
    plt.close(figure)


def save_figure(
    figure: plt.Figure, base_name: str, exclude_axes: list[plt.Axes] | None = None
) -> None:
    # 图上的内容先画出来，再按原有流程看一下各个面板有没有对齐。
    # 这一关通过以后再保存图片，保存好了以后就关闭当前图，后面继续画其他的
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    with TemporaryDirectory(prefix="problem3-figure-qa-") as tmp_dir:
        require_matplotlib_panel_alignment(
            figure,
            json_out=Path(tmp_dir) / f"{base_name}.alignment.json",
            exclude_axes=exclude_axes or [],
            row_groups=getattr(figure, "_alignment_row_groups", None),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    figure.savefig(
        figure_dir / f"{base_name}.png", dpi=export_dpi, bbox_inches="tight",
        pad_inches=pad_inches, facecolor="white",
    )
    plt.close(figure)


def plot_plateau(data: dict[str, object]) -> plt.Figure:
    # 把末尾那一段实测点先画出来，均值也在同一块位置上接着画。
    # 两种信息一起留在图里，后面查看平台段的时候就可以一起看
    times = np.asarray(data["boundary_times_h"])
    temp = np.asarray(data["room_temperature"])
    water = np.asarray(data["room_moisture"])
    stable = times >= times[-1] - 1.0
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    series = ((temp, "温度 / ℃", color_teal), (water, "环境含水率 / kg·kg$^{-1}$", color_coral))
    for ax, (values, ylabel, color) in zip(axs, series):
        ax.plot(times[stable], values[stable], "o", ms=2.6, color=color, alpha=0.75)
        ax.axhline(values[stable].mean(), color=color_plum, lw=1.5)
        ax.set(xlabel="时间 / h", ylabel=ylabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        style_axis(ax)
    axs[0].set_title("末 1 h 温度稳定段", pad=7)
    axs[1].set_title("末 1 h 水分边界稳定段", pad=7)
    fig.text(0.5, 0.845, "圆点为实测值，实线为末 1 h 均值", ha="center", color=color_dark, fontsize=7.3)
    set_title(fig, "长期边界采用末 1 h 稳定平台", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.73, wspace=0.34)
    return fig


def plot_threshold(data: dict[str, object]) -> plt.Figure:
    # 先把中心和表面的变化放出来，让时间曲线从头到尾接起来。
    # 达标附近还要再单独看一看，所以相关的时间记录也一起带到这一张图里。
    # 整段曲线先留着，到了要看细节的地方，再对着对应的时间记录找过去
    times = np.asarray(data["times_h"])
    center = np.asarray(data["center_moisture"])
    surface = np.asarray(data["surface_moisture"])
    target = float(data["continuous_threshold_time_h"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axs[0].plot(times, center, color=color_teal, lw=1.7, label="轴心（全域最大值）")
    axs[0].plot(times, surface, color=color_coral, lw=1.35, label="表面")
    axs[0].axhline(default_critical_moisture, color=color_dark, ls="--", lw=1.0, label="达标阈值")
    axs[0].set(xlabel="时间 / h", ylabel="含水率 / kg·kg$^{-1}$", title="全过程")
    zoom = (times >= target - 0.22) & (times <= target + 0.18)
    axs[1].plot(times[zoom], center[zoom], "o-", ms=2.3, color=color_teal, lw=1.25)
    axs[1].axhline(default_critical_moisture, color=color_dark, ls="--", lw=1.0)
    axs[1].axvline(target, color=color_coral, lw=1.35)
    axs[1].annotate(
        f"$t_*= {target:.4f}$ h", xy=(target, default_critical_moisture),
        xycoords="data", xytext=(0.75, 0.78), textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": color_coral, "lw": 0.8},
        color=color_coral, fontsize=7.5, ha="center", va="center", zorder=5,
    )
    axs[1].set(xlabel="时间 / h", ylabel="轴心含水率 / kg·kg$^{-1}$", title="阈值附近")
    axs[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axs:
        style_axis(ax)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=3)
    set_title(fig, "轴心控制全域达标时间：57.52 h", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.70, wspace=0.34)
    return fig


def plot_field(data: dict[str, object]) -> plt.Figure:
    # 不想写了
    times = np.asarray(data["times_h"])
    radii = np.asarray(data["radii_cm"])
    water = np.asarray(data["moisture"])
    discrete = float(data["discrete_threshold_time_h"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.35))
    mesh = axs[0].pcolormesh(times, radii, water.T, shading="auto", cmap="viridis")
    axs[0].axvline(discrete, color="white", ls="--", lw=1.0)
    axs[0].set(xlabel="时间 / h", ylabel="径向位置 / cm", title="时空分布")
    color_axis = axs[0].inset_axes([1.025, 0.02, 0.045, 0.96])
    fig.colorbar(mesh, cax=color_axis)
    for label in color_axis.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    selected = (0.0, 6.0, 12.0, 24.0, 36.0, discrete)
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.90, len(selected)))
    for hour, color in zip(selected, colors):
        index = int(np.argmin(np.abs(times - hour)))
        label = f"{times[index]:.1f} h" if hour else "0 h"
        axs[1].plot(radii, water[index], color=color, lw=1.45, label=label)
    axs[1].axhline(default_critical_moisture, color=color_dark, ls="--", lw=0.9)
    axs[1].set(xlabel="径向位置 / cm", ylabel="含水率 / kg·kg$^{-1}$", title="代表时刻径向剖面")
    style_axis(axs[0], show_grid=False)
    style_axis(axs[1])
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=6)
    set_title(fig, "含水率由表面向轴心逐步衰减", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.69, wspace=0.40)
    fig._alignment_row_groups = [["a", "b"]]
    fig._extra_qa_axes = [color_axis]
    return fig


def plot_tail(data: dict[str, object]) -> plt.Figure:
    # 后期变化比较慢的那部分，这里再单独拿出来放大看一下。
    # 要用的曲线和数值先分别取好，再一起排到这一张图上，查看时就不用来回换图，方便
    times = np.asarray(data["times_h"])
    center = np.asarray(data["center_moisture"])
    stats = data["diagnostics"]
    plateau_temp = float(stats["baseline"]["plateau_temperature_c"])
    diffusivity = problem3_solver.moisture_diffusivity(center, np.full_like(center, plateau_temp))
    early_rate, late_rate = phase_rates(data)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axs[0].plot(times, center, color=color_teal, lw=1.7)
    axs[0].axhline(default_critical_moisture, color=color_dark, ls="--", lw=0.9)
    axs[0].axvspan(0.0, 12.0, color=color_sand, alpha=0.48)
    axs[0].axvspan(36.0, times[-1], color=color_coral, alpha=0.10)
    axs[0].set(xlabel="时间 / h", ylabel="轴心含水率 / kg·kg$^{-1}$", title="两阶段干燥过程")
    axs[1].semilogy(times, diffusivity / diffusivity[0], color=color_plum, lw=1.7)
    axs[1].set(xlabel="时间 / h", ylabel="相对扩散系数 $D/D_0$", title="低含水率下扩散能力衰减")
    axs[1].text(
        0.97, 0.93, f"初期 {early_rate:.4f}\n后期 {late_rate:.4f}\n速率相差 >100 倍",
        transform=axs[1].transAxes, ha="right", va="top", color=color_dark,
        fontsize=7.4, linespacing=1.35,
    )
    for ax in axs:
        style_axis(ax)
    set_title(fig, "扩散系数退化导致后期拖尾", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.74, wspace=0.34)
    return fig


def plot_lollipop(axis: plt.Axes, labels: list[str], values: np.ndarray, xlabel: str) -> None:
    # 每一种情况先占一行，横线从零的位置接到对应的数，再在那头放个点。
    # 标签和旁边的数也一起跟上，这样一行一行往下看就能对起来。
    # 都放好以后，再把坐标轴周围整理一下。
    positions = np.arange(len(values))
    colors = [color_coral if value > 0 else color_teal for value in values]
    axis.hlines(positions, 0.0, values, color=colors, lw=2.0)
    axis.scatter(values, positions, color=colors, s=27, zorder=3)
    axis.axvline(0.0, color=color_dark, lw=0.8)
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
    style_axis(axis)


def plot_robustness(data: dict[str, object]) -> plt.Figure:
    # 几种计算设置的结果现在放到一起，顺着标签一个一个往下排。
    # 每一种变化了多少都用已有的记录来展示
    stats = data["diagnostics"]
    baseline = float(stats["baseline"]["continuous_threshold_time_h"])
    boundary_cases = stats["boundary_scenarios"]
    boundary_labels = ["末 30 min", "最后采样点", "温度 -1σ", "温度 +1σ", "水分 -1σ", "水分 +1σ"]
    boundary_offsets = 60.0 * np.asarray([case["continuous_threshold_time_h"] - baseline for case in boundary_cases])
    refinements = stats["numerical_refinement"]
    numerical_labels = ["空间加密", "时间加密", "端面暴露"]
    numerical_offsets = np.asarray([
        3600.0 * (refinements[0]["continuous_threshold_time_h"] - baseline),
        3600.0 * (refinements[1]["continuous_threshold_time_h"] - baseline),
        -7.6,
    ])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.45))
    plot_lollipop(axs[0], boundary_labels, boundary_offsets, "相对基准达标时间变化 / min")
    plot_lollipop(axs[1], numerical_labels, numerical_offsets, "达标时间变化 / s")
    axs[0].set_title("长期边界情景", pad=7)
    axs[1].set_title("离散与端面检验", pad=7)
    set_title(fig, "主要结论对边界扰动与数值设置稳定", y=0.985)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.74, wspace=0.58)
    return fig


def clear_old_figures() -> None:
    # 新图的名字先记下来，再看目录里还有哪些图片不在这份名单里。
    figure_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{name}.png" for name in new_figure_bases}
    for target in figure_dir.iterdir():
        if target.is_file() and target.suffix.lower() in {".png", ".pdf", ".svg", ".tif", ".tiff"} and target.name not in expected:
            if target.resolve().parent != figure_dir.resolve():
                raise RuntimeError("拒绝删除问题三图片目录之外的文件。")
            target.unlink()


def export_figures(data: dict[str, object]) -> None:
    clear_old_figures()
    builders = (plot_plateau, plot_threshold, plot_field, plot_tail, plot_robustness)
    for builder, base_name in zip(builders, new_figure_bases):
        fig = builder(data)
        save_figure(fig, base_name, getattr(fig, "_extra_qa_axes", None))


def main() -> None:
    # 这个就不写注释了
    export_figures(load_plot_data())
    print(f"问题三绘图完成，共生成 {len(new_figure_bases)} 张 600 dpi PNG：{figure_dir}")


if __name__ == "__main__":
    main()