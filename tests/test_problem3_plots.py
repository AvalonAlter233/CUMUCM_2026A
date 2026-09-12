from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import 问题三_绘图 as plotting


def _has_visible_major_grid(axis):
    return (
        any(line.get_visible() for line in axis.get_xgridlines())
        and any(line.get_visible() for line in axis.get_ygridlines())
    )


def test_publication_contract_is_five_png_figures_at_600_dpi():
    assert len(plotting.NEW_FIGURE_BASES) == 5
    assert len(set(plotting.NEW_FIGURE_BASES)) == 5
    assert plotting.FIGURE_FORMATS == ("png",)
    assert plotting.EXPORT_DPI == 600
    assert plotting.CJK_FONT_CANDIDATES[0] == "STSong"
    assert plotting.mpl.rcParams["font.family"] == ["serif"]
    assert plotting.mpl.rcParams["font.serif"][:2] == ["STSong", "SimSun"]
    assert plotting.mpl.rcParams["mathtext.fontset"] == "stix"
    assert plotting.mpl.rcParams["axes.titleweight"] == "bold"
    assert plotting.mpl.rcParams["axes.labelweight"] == "bold"
    assert plotting.mpl.rcParams["svg.fonttype"] == "none"
    assert plotting.mpl.rcParams["pdf.fonttype"] == 42
    assert plotting.REFERENCE_PALETTE == (
        "#44757A", "#452A3D", "#D44C3C", "#EED5B7"
    )


def test_academic_font_renders_chinese_without_missing_glyph_warnings():
    figure, axis = plt.subplots()
    axis.set_title("问题三含水率，温度：50 ℃；参数 σ·φξ→达标")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.canvas.draw()
    assert not [warning for warning in caught if "missing from font" in str(warning.message)]
    plotting.close_without_export(figure)


def test_loaded_result_matches_reported_threshold_and_center_control():
    data = plotting.load_plot_data()
    assert data["continuous_threshold_time_h"] == pytest.approx(57.5223926)
    assert data["discrete_threshold_time_h"] == pytest.approx(57.5333333)
    np.testing.assert_allclose(
        data["maximum_moisture"], data["center_moisture"], atol=1e-12
    )


def test_reported_drying_rates_show_two_order_tail_slowdown():
    data = plotting.load_plot_data()
    early_rate, late_rate = plotting.drying_phase_rates(data)
    assert early_rate == pytest.approx(0.1744, rel=0.01)
    assert late_rate == pytest.approx(0.0016, rel=0.08)
    assert early_rate / late_rate > 100.0


def test_core_figures_are_compact_and_use_moisture_heatmap_palette():
    data = plotting.load_plot_data()
    target = plotting.plot_threshold_evidence(data)
    field = plotting.plot_field_evolution(data)
    assert len(target.axes) == 2
    assert field.axes[0].collections[0].cmap.name == "viridis"
    assert field.axes[0].child_axes[0].get_ylabel() == ""
    assert len(field.legends) == 1
    plotting.close_without_export(target)
    plotting.close_without_export(field)


def test_threshold_annotation_uses_clear_upper_right_axes_position():
    figure = plotting.plot_threshold_evidence(plotting.load_plot_data())
    annotation = next(
        text for text in figure.axes[1].texts if "t_*" in text.get_text()
    )
    assert annotation.xycoords == "data"
    assert annotation._textcoords == "axes fraction"
    text_x, text_y = annotation.get_position()
    assert text_x > 0.60
    assert text_y > 0.65
    assert annotation.get_bbox_patch() is None
    plotting.close_without_export(figure)


def test_all_non_heatmap_axes_show_grids_while_heatmap_axis_does_not():
    data = plotting.load_plot_data()
    ordinary_builders = (
        plotting.plot_boundary_plateau,
        plotting.plot_threshold_evidence,
        plotting.plot_drying_tail,
        plotting.plot_robustness,
    )
    for builder in ordinary_builders:
        figure = builder(data)
        assert all(_has_visible_major_grid(axis) for axis in figure.axes)
        plotting.close_without_export(figure)

    field = plotting.plot_field_evolution(data)
    assert not _has_visible_major_grid(field.axes[0])
    assert _has_visible_major_grid(field.axes[1])
    plotting.close_without_export(field)


def test_robustness_value_labels_stay_to_the_right_of_markers():
    figure = plotting.plot_robustness(plotting.load_plot_data())
    assert all(text.get_ha() == "left" for axis in figure.axes for text in axis.texts)
    assert all(text.get_fontsize() >= 7.2 for axis in figure.axes for text in axis.texts)
    plotting.close_without_export(figure)


def test_all_figures_have_centered_bold_titles_without_panel_letters():
    data = plotting.load_plot_data()
    builders = (
        plotting.plot_boundary_plateau,
        plotting.plot_threshold_evidence,
        plotting.plot_field_evolution,
        plotting.plot_drying_tail,
        plotting.plot_robustness,
    )
    for builder in builders:
        figure = builder(data)
        assert figure._suptitle is not None
        assert figure._suptitle.get_position()[0] == pytest.approx(0.5)
        assert figure._suptitle.get_fontweight() == "bold"
        assert figure._suptitle.get_fontsize() >= 12
        texts = [text.get_text().strip().lower() for axis in figure.axes for text in axis.texts]
        assert "a" not in texts and "b" not in texts
        plotting.close_without_export(figure)


def test_main_exports_only_the_new_png_set(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "FIGURE_DIR", Path(tmp_path))
    (Path(tmp_path) / "旧图.png").write_bytes(b"legacy")
    plotting.main()
    figures = sorted(path.name for path in Path(tmp_path).glob("*.png"))
    assert figures == sorted(f"{name}.png" for name in plotting.NEW_FIGURE_BASES)
    assert all((Path(tmp_path) / name).stat().st_size > 10_000 for name in figures)
    assert not list(Path(tmp_path).glob("*.pdf"))
    assert not list(Path(tmp_path).glob("*.svg"))
    plt.close("all")
