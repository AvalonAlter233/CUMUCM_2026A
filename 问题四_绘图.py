"""问题四论文配图：收缩自洽性、移动域演化、机制交互与稳健性。

主图数据来自 result4.xlsx、内部细网格和已完成复算的诊断文件；
脚本只导出 600 dpi PNG。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

import 问题4_求解 as problem4


NATURE_FIGURE_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(NATURE_FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NATURE_FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result4.xlsx"
DIAGNOSTICS_FILE = PROJECT_ROOT / "附件" / "附件3" / "result4_diagnostics.json"
INTERNAL_FIELD_FILE = PROJECT_ROOT / "附件" / "附件3" / "result4_internal_field.npz"
FIGURE_DIR = PROJECT_ROOT / "figures" / "问题四"

CRITICAL_MOISTURE = 0.15
INITIAL_MOISTURE = 2.55
INITIAL_RADIUS_CM = 2.0
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
    "图1_收缩轨迹与含水率自洽性",
    "图2_移动域达标时间判定",
    "图3_移动域含水率演化",
    "图4_物性与几何非线性交互",
    "图5_数值与边界稳健性",
)


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
    with np.load(path) as data:
        times_h = np.asarray(data["times_s"], dtype=float) / 3600.0
        xi_centers = np.asarray(data["xi_centers"], dtype=float)
        moisture = np.asarray(data["moisture"], dtype=float)
        surface_radii_cm = 100.0 * np.asarray(data["surface_radii_m"], dtype=float)
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
    if len(indices) == 0 or indices[0] == 0:
        raise ValueError("输出记录没有形成有效的阈值夹逼。")
    index = int(indices[0])
    y0, y1 = float(maximum_moisture[index - 1]), float(maximum_moisture[index])
    crossing = float(times_h[index - 1]) + (y0 - threshold) / (y0 - y1) * float(
        times_h[index] - times_h[index - 1]
    )
    return crossing, float(times_h[index])


def verified_mechanism_cases(diagnostics: dict) -> list[dict]:
    if not diagnostics.get("verification_complete", False):
        return []
    cases_by_name = {
        case["name"]: case
        for case in [diagnostics["baseline"], *diagnostics["mechanism_comparison"]]
    }
    ordered_names = (
        "appendix3_fixed_radius", "appendix3_moving_radius",
        "appendix4_fixed_radius", "appendix4_moving_radius",
    )
    return [cases_by_name[name] for name in ordered_names]


def load_plot_data() -> dict[str, object]:
    output_times_h, fixed_radii_cm, output_moisture = read_result()
    internal_times_h, xi_centers, internal_moisture, surface_radii_cm = read_internal_field()
    if not np.allclose(output_times_h, internal_times_h, rtol=0.0, atol=1e-12):
        raise ValueError("内部细网格与结果工作簿的输出时刻不一致。")
    with DIAGNOSTICS_FILE.open("r", encoding="utf-8") as stream:
        diagnostics = json.load(stream)
    history = problem4.read_radius_history(problem4.RADIUS_FILE)

    times_h = np.concatenate(([0.0], output_times_h))
    output = np.vstack((np.full((1, output_moisture.shape[1]), INITIAL_MOISTURE), output_moisture))
    internal = np.vstack((np.full((1, internal_moisture.shape[1]), INITIAL_MOISTURE), internal_moisture))
    surface_radius = np.concatenate(([INITIAL_RADIUS_CM], surface_radii_cm))
    maximum = np.maximum(output.max(axis=1), internal.max(axis=1))
    baseline = diagnostics["baseline"]
    return {
        "times_h": times_h,
        "fixed_radii_cm": fixed_radii_cm,
        "output_moisture": output,
        "xi_centers": xi_centers,
        "internal_moisture": internal,
        "surface_radii_cm": surface_radius,
        "center_moisture": output[:, 0],
        "surface_moisture": output[:, -1],
        "maximum_moisture": maximum,
        "radius_history_times_h": history.times / 3600.0,
        "radius_history_cm": history.radii * 100.0,
        "continuous_threshold_time_h": float(baseline["continuous_threshold_time_h"]),
        "discrete_threshold_time_h": float(baseline["discrete_threshold_time_h"]),
        "diagnostics": diagnostics,
    }


def shrinkage_driver_fields(data: dict[str, object]) -> dict[str, np.ndarray]:
    xi = np.asarray(data["xi_centers"], dtype=float)
    moisture = np.asarray(data["internal_moisture"], dtype=float)
    surface = np.asarray(data["surface_moisture"], dtype=float)
    volume_average = 2.0 * np.trapezoid(moisture * xi[None, :], xi, axis=1)
    outer_mask = xi >= 0.5
    outer_xi = xi[outer_mask]
    outer_average = np.trapezoid(
        moisture[:, outer_mask] * outer_xi[None, :], outer_xi, axis=1
    ) / np.trapezoid(outer_xi, outer_xi)
    return {
        "volume_average": volume_average,
        "outer_average": outer_average,
        "surface": surface,
    }


def shrinkage_phi(data: dict[str, object]) -> dict[str, np.ndarray]:
    radius = np.asarray(data["surface_radii_cm"], dtype=float)
    s = (radius / INITIAL_RADIUS_CM) ** 2
    denominator = 1.0 - s
    fields = shrinkage_driver_fields(data)
    result = {}
    for name, values in fields.items():
        phi = np.full_like(values, np.nan)
        valid = denominator > 1e-8
        phi[valid] = (s[valid] * INITIAL_MOISTURE - values[valid]) / denominator[valid]
        result[name] = phi
    return result


def shrinkage_consistency_statistics(
    data: dict[str, object], start_h: float = 12.0
) -> dict[str, dict[str, float]]:
    times = np.asarray(data["times_h"], dtype=float)
    trajectories = shrinkage_phi(data)
    boundaries = ((start_h, 24.0), (24.0, 40.0), (40.0, np.inf))
    statistics = {}
    for name, values in trajectories.items():
        selected = (times >= start_h) & np.isfinite(values)
        mean = float(np.mean(values[selected]))
        squared_residual = 0.0
        count = 0
        for lower, upper in boundaries:
            mask = selected & (times >= lower) & (times < upper)
            if np.any(mask):
                local = values[mask]
                squared_residual += float(np.sum((local - np.mean(local)) ** 2))
                count += int(mask.sum())
        pooled_std = float(np.sqrt(squared_residual / count))
        statistics[name] = {
            "mean": mean,
            "within_segment_std": pooled_std,
            "coefficient_of_variation": pooled_std / abs(mean),
        }
    return statistics


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
    with TemporaryDirectory(prefix="problem4-figure-qa-") as temporary_dir:
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


def plot_shrinkage_consistency(data: dict[str, object]) -> plt.Figure:
    history_times = np.asarray(data["radius_history_times_h"])
    history_radius = np.asarray(data["radius_history_cm"])
    times = np.asarray(data["times_h"])
    phi = shrinkage_phi(data)
    statistics = shrinkage_consistency_statistics(data)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))

    axes[0].plot(history_times, history_radius, "o-", ms=2.0, lw=1.35, color=COLOR_TEAL)
    axes[0].scatter([history_times[0], history_times[-1]], [history_radius[0], history_radius[-1]], color=COLOR_CORAL, s=24, zorder=3)
    axes[0].set(xlabel="时间 / h", ylabel="药材半径 / cm", title="附件二实测收缩轨迹")
    axes[0].text(0.96, 0.92, "2.000 → 1.198 cm\n径向收缩 40.1%", transform=axes[0].transAxes, ha="right", va="top", color=COLOR_DARK)

    mask = (times >= 6.0) & np.isfinite(phi["surface"])
    for name, label, color, width in (
        ("volume_average", "体积平均", COLOR_SAND, 1.3),
        ("outer_average", "外层半域平均", COLOR_PLUM, 1.3),
        ("surface", "表面含水率", COLOR_CORAL, 1.8),
    ):
        axes[1].plot(times[mask], phi[name][mask], color=color, lw=width, label=label)
    axes[1].axvline(12.0, color=COLOR_DARK, ls="--", lw=0.8)
    axes[1].axhline(statistics["surface"]["mean"], color=COLOR_CORAL, ls=":", lw=1.0)
    axes[1].set(xlabel="时间 / h", ylabel=r"反演参数 $\phi$", title="不同收缩驱动量的恒定性")
    axes[1].set_ylim(0.15, 1.62)
    axes[1].text(0.97, 0.10, "12 h 后表面口径 CV = 0.16%", transform=axes[1].transAxes, ha="right", color=COLOR_CORAL, fontsize=7.3)
    for axis in axes:
        _style_axis(axis)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=3)
    _set_top_title(figure, "收缩轨迹与表面含水率保持自洽", y=0.985)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.70, wspace=0.34)
    return figure


def plot_threshold_evidence(data: dict[str, object]) -> plt.Figure:
    times = np.asarray(data["times_h"])
    center = np.asarray(data["center_moisture"])
    surface = np.asarray(data["surface_moisture"])
    target = float(data["continuous_threshold_time_h"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axes[0].plot(times, center, color=COLOR_TEAL, lw=1.7, label="轴心（全域最大值）")
    axes[0].plot(times, surface, color=COLOR_CORAL, lw=1.35, label="移动表面")
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
    _set_top_title(figure, "移动边界模型全域达标时间：51.12 h", y=0.985)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.70, wspace=0.34)
    return figure


def plot_moving_field(data: dict[str, object]) -> plt.Figure:
    times = np.asarray(data["times_h"])
    xi = np.asarray(data["xi_centers"])
    internal = np.asarray(data["internal_moisture"])
    radii = np.asarray(data["surface_radii_cm"])
    center = np.asarray(data["center_moisture"])
    surface = np.asarray(data["surface_moisture"])
    discrete = float(data["discrete_threshold_time_h"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    mesh = axes[0].pcolormesh(times, xi, internal.T, shading="auto", cmap="viridis")
    axes[0].axvline(discrete, color="white", ls="--", lw=1.0)
    axes[0].set(xlabel="时间 / h", ylabel=r"归一化半径 $\xi=r/R(t)$", title="移动参考域时空分布")
    color_axis = axes[0].inset_axes([1.025, 0.02, 0.045, 0.96])
    figure.colorbar(mesh, cax=color_axis)

    selected = (0.0, 6.0, 12.0, 24.0, 36.0, discrete)
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.90, len(selected)))
    for hour, color in zip(selected, colors):
        index = int(np.argmin(np.abs(times - hour)))
        physical_radii = np.concatenate(([0.0], xi * radii[index], [radii[index]]))
        profile = np.concatenate(([center[index]], internal[index], [surface[index]]))
        label = f"{times[index]:.1f} h" if hour else "0 h"
        axes[1].plot(physical_radii, profile, color=color, lw=1.45, label=label)
    axes[1].axhline(CRITICAL_MOISTURE, color=COLOR_DARK, ls="--", lw=0.9)
    axes[1].set(xlabel="实际径向位置 / cm", ylabel="含水率 / kg·kg$^{-1}$", title="收缩中的径向剖面")
    for axis in axes:
        _style_axis(axis)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=6)
    _set_top_title(figure, "收缩缩短扩散路径并保持轴心最湿", y=0.985)
    figure.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.69, wspace=0.40)
    figure._alignment_row_groups = [["a", "b"]]
    figure._extra_qa_axes = [color_axis]
    return figure


def plot_mechanism_interaction(data: dict[str, object]) -> plt.Figure:
    cases = verified_mechanism_cases(data["diagnostics"])
    if not cases:
        raise ValueError("机制对照尚未完成整套复算。")
    values = {case["name"]: float(case["continuous_threshold_time_h"]) for case in cases}
    figure, axis = plt.subplots(figsize=(5.5, 3.35))
    x = np.arange(2)
    appendix3 = [values["appendix3_fixed_radius"], values["appendix3_moving_radius"]]
    appendix4 = [values["appendix4_fixed_radius"], values["appendix4_moving_radius"]]
    axis.plot(x, appendix3, "o-", color=COLOR_TEAL, lw=1.8, ms=5, label="附录三物性")
    axis.plot(x, appendix4, "o-", color=COLOR_CORAL, lw=1.8, ms=5, label="附录四物性")
    axis.set_xticks(x, ["固定半径", "实测收缩半径"])
    axis.set(ylabel="连续达标时间 $t_*$ / h", title="两条非平行响应线表明物性与几何存在交互")
    for series, color in ((appendix3, COLOR_TEAL), (appendix4, COLOR_CORAL)):
        for px, value in zip(x, series):
            axis.text(px, value + 3.0, f"{value:.2f}", ha="center", color=color, fontsize=7.4)
    _style_axis(axis)
    figure.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=2)
    _set_top_title(figure, "物性变化与收缩效应不可简单相加", y=0.985)
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.68)
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
    refinement = diagnostics["numerical_refinement"]
    refinement_offsets = 3600.0 * np.asarray([case["continuous_threshold_time_h"] - baseline for case in refinement])
    boundary = diagnostics["boundary_sensitivity"]
    boundary_offsets = 60.0 * np.asarray([case["continuous_threshold_time_h"] - baseline for case in boundary])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    _lollipop(axes[0], ["空间加密", "时间加密"], refinement_offsets, "达标时间变化 / s")
    _lollipop(axes[1], ["温度 -1σ", "温度 +1σ"], boundary_offsets, "相对基准变化 / min")
    axes[0].set_title("网格与步长独立性", pad=7)
    axes[1].set_title("长期温度边界敏感性", pad=7)
    _set_top_title(figure, "51.12 h 结论在数值与边界检验下稳定", y=0.985)
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.73, wspace=0.52)
    return figure


def remove_legacy_outputs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    expected = {f"{name}.png" for name in NEW_FIGURE_BASES}
    for target in FIGURE_DIR.iterdir():
        if target.is_file() and target.suffix.lower() in {".png", ".pdf", ".svg", ".tif", ".tiff"} and target.name not in expected:
            if target.resolve().parent != FIGURE_DIR.resolve():
                raise RuntimeError("拒绝删除问题四图片目录之外的文件。")
            target.unlink()


def export_all(data: dict[str, object]) -> None:
    remove_legacy_outputs()
    builders = (
        plot_shrinkage_consistency, plot_threshold_evidence, plot_moving_field,
        plot_mechanism_interaction, plot_robustness,
    )
    for builder, base_name in zip(builders, NEW_FIGURE_BASES):
        figure = builder(data)
        save_publication_figure(figure, base_name, getattr(figure, "_extra_qa_axes", None))


def main() -> None:
    export_all(load_plot_data())
    print(f"问题四绘图完成，共生成 {len(NEW_FIGURE_BASES)} 张 600 dpi PNG：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
