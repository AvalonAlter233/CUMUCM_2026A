from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import 问题四_绘图 as plotting


def _has_visible_major_grid(axis):
    return (
        any(line.get_visible() for line in axis.get_xgridlines())
        and any(line.get_visible() for line in axis.get_ygridlines())
    )


def test_result_reader_and_threshold_detection_match_workbook():
    times, radii, moisture = plotting.read_result()
    assert times[0] == pytest.approx(60.0 / 3600.0)
    assert times[-1] == pytest.approx(51.11666666666667)
    assert np.allclose(radii, np.arange(0.0, 1.2, 0.1))
    assert moisture.shape == (3067, 13)
    continuous, discrete = plotting.threshold_times_from_output(
        times, moisture.max(axis=1)
    )
    assert continuous < discrete
    assert discrete == pytest.approx(51.11666666666667)


def test_internal_field_reader_returns_saved_reference_grid(tmp_path):
    path = tmp_path / "internal_field.npz"
    np.savez_compressed(
        path,
        times_s=np.array([60.0, 120.0]),
        xi_centers=np.array([0.25, 0.75]),
        moisture=np.array([[2.0, 1.0], [1.8, 0.8]]),
        surface_radii_m=np.array([0.019, 0.018]),
    )
    times_h, xi, moisture, radii_cm = plotting.read_internal_field(path)
    np.testing.assert_allclose(times_h, [1.0 / 60.0, 2.0 / 60.0])
    np.testing.assert_allclose(xi, [0.25, 0.75])
    np.testing.assert_allclose(moisture, [[2.0, 1.0], [1.8, 0.8]])
    np.testing.assert_allclose(radii_cm, [1.9, 1.8])


def test_incomplete_diagnostics_do_not_produce_mechanism_plot():
    diagnostics = {
        "verification_complete": False,
        "baseline": {"name": "appendix4_moving_radius"},
        "mechanism_comparison": [{"name": "stale_case"}],
    }
    assert plotting.verified_mechanism_cases(diagnostics) == []


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


def test_load_plot_data_matches_reported_threshold_and_center_control():
    data = plotting.load_plot_data()
    assert data["continuous_threshold_time_h"] == pytest.approx(51.1151669)
    assert data["discrete_threshold_time_h"] == pytest.approx(51.1166667)
    np.testing.assert_allclose(
        data["maximum_moisture"], data["center_moisture"], atol=1e-12
    )


def test_shrinkage_consistency_reproduces_paper_statistics():
    data = plotting.load_plot_data()
    statistics = plotting.shrinkage_consistency_statistics(data, start_h=12.0)
    assert statistics["volume_average"]["mean"] == pytest.approx(1.1505, rel=0.01)
    assert statistics["outer_average"]["mean"] == pytest.approx(1.1809, rel=0.01)
    assert statistics["surface"]["mean"] == pytest.approx(1.3485, rel=0.003)
    assert statistics["surface"]["coefficient_of_variation"] < 0.002


def test_core_figures_show_moving_field_and_nonparallel_interaction():
    data = plotting.load_plot_data()
    field = plotting.plot_moving_field(data)
    interaction = plotting.plot_mechanism_interaction(data)
    assert field.axes[0].collections[0].cmap.name == "viridis"
    assert field.axes[0].child_axes[0].get_ylabel() == ""
    assert len(field.legends) == 1
    assert len(interaction.axes) == 1
    assert len(interaction.axes[0].lines) == 2
    y0 = interaction.axes[0].lines[0].get_ydata()
    y1 = interaction.axes[0].lines[1].get_ydata()
    assert not np.isclose(y0[1] - y0[0], y1[1] - y1[0])
    plotting.close_without_export(field)
    plotting.close_without_export(interaction)


def test_all_non_heatmap_axes_show_grids_while_heatmap_axis_does_not():
    data = plotting.load_plot_data()
    ordinary_builders = (
        plotting.plot_shrinkage_consistency,
        plotting.plot_threshold_evidence,
        plotting.plot_mechanism_interaction,
        plotting.plot_robustness,
    )
    for builder in ordinary_builders:
        figure = builder(data)
        assert all(_has_visible_major_grid(axis) for axis in figure.axes)
        plotting.close_without_export(figure)

    field = plotting.plot_moving_field(data)
    assert not _has_visible_major_grid(field.axes[0])
    assert _has_visible_major_grid(field.axes[1])
    plotting.close_without_export(field)


def test_robustness_value_labels_stay_to_the_right_of_markers():
    figure = plotting.plot_robustness(plotting.load_plot_data())
    assert all(text.get_ha() == "left" for axis in figure.axes for text in axis.texts)
    plotting.close_without_export(figure)


def test_all_figures_have_centered_bold_titles_without_panel_letters():
    data = plotting.load_plot_data()
    builders = (
        plotting.plot_shrinkage_consistency,
        plotting.plot_threshold_evidence,
        plotting.plot_moving_field,
        plotting.plot_mechanism_interaction,
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


def test_plotting_main_generates_only_five_nonempty_png_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "FIGURE_DIR", Path(tmp_path))
    (Path(tmp_path) / "旧图.png").write_bytes(b"legacy")
    plotting.main()
    figures = sorted(path.name for path in Path(tmp_path).glob("*.png"))
    assert figures == sorted(f"{name}.png" for name in plotting.NEW_FIGURE_BASES)
    assert all((Path(tmp_path) / name).stat().st_size > 10_000 for name in figures)
    assert not list(Path(tmp_path).glob("*.pdf"))
    assert not list(Path(tmp_path).glob("*.svg"))
    plt.close("all")
