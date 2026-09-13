import shutil

import numpy as np
import pytest

import 问题二_绘图 as plotting


def synthetic_plot_data():
    times_s = np.array([1.0, 1800.0, 3600.0, 7200.0, 10800.0])
    radii_cm = np.array([0.0, 1.0, 2.0])
    temperature = np.array(
        [
            [28.0, 28.1, 28.2],
            [32.2, 33.0, 35.4],
            [40.4, 41.1, 43.0],
            [48.4, 48.6, 49.0],
            [49.85, 49.87, 49.97],
        ]
    )
    moisture = np.array(
        [
            [2.55, 2.54, 2.50],
            [2.55, 2.53, 1.65],
            [2.53, 2.36, 1.47],
            [2.17, 1.92, 1.23],
            [1.77, 1.57, 1.01],
        ]
    )
    q1_times_s = np.array([1.0, 900.0, 1800.0])
    q1_temperature = np.array(
        [[28.0, 28.1, 28.2], [29.7, 30.4, 32.1], [33.58, 34.7, 36.79]]
    )
    q1_moisture = np.array(
        [[2.55, 2.54, 2.50], [2.55, 2.51, 1.90], [2.54, 2.35, 1.51]]
    )
    return {
        "times_s": times_s,
        "radii_cm": radii_cm,
        "temperature_field": temperature,
        "moisture_field": moisture,
        "average_temperature": plotting.radial_area_average(radii_cm, temperature),
        "average_moisture": plotting.radial_area_average(radii_cm, moisture),
        "q1_times_s": q1_times_s,
        "q1_temperature_field": q1_temperature,
        "q1_moisture_field": q1_moisture,
    }


def test_prepend_initial_state_adds_t_zero():
    times = np.array([1.0, 2.0])
    field = np.array([[1.5, 1.0], [1.4, 0.9]])

    new_times, new_field = plotting.prepend_initial_state(times, field, 2.0)

    np.testing.assert_allclose(new_times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(new_field[0], [2.0, 2.0])
    np.testing.assert_allclose(new_field[1:], field)


def test_radial_area_average_uses_cylindrical_weighting():
    radii = np.array([0.0, 1.0, 2.0])
    uniform = np.array([[3.0, 3.0, 3.0], [1.5, 1.5, 1.5]])

    np.testing.assert_allclose(plotting.radial_area_average(radii, uniform), [3.0, 1.5])

    with pytest.raises(ValueError, match="严格递增"):
        plotting.radial_area_average(np.array([0.0, 1.0, 1.0]), uniform)


def test_new_figure_contract_contains_exactly_five_png_outputs():
    assert len(plotting.NEW_FIGURE_BASES) == 5
    assert len(set(plotting.NEW_FIGURE_BASES)) == 5
    assert plotting.FIGURE_FORMATS == ("png",)
    assert plotting.EXPORT_DPI == 600


def test_style_uses_song_fonts_and_visible_paper_grid():
    assert plotting.CJK_FONT_CANDIDATES[0] == "STSong"
    assert plotting.mpl.rcParams["font.family"] == ["serif"]
    assert plotting.mpl.rcParams["font.serif"][:2] == ["STSong", "SimSun"]
    assert plotting.HEADING_FONT == "STZhongsong"
    figure, axis = plotting.plt.subplots()
    plotting._style_axis(axis)
    line = next(line for line in axis.get_ygridlines() if line.get_visible())
    assert line.get_color().upper() == "#AAA5A8"
    assert line.get_linewidth() == pytest.approx(0.55)
    assert line.get_alpha() == pytest.approx(0.70)
    plotting.close_without_export(figure)


def test_reference_palette_and_heatmap_colormaps_are_preserved():
    assert plotting.REFERENCE_PALETTE == (
        "#44757A", "#452A3D", "#D44C3C", "#EED5B7"
    )
    figure = plotting.plot_field_evolution(synthetic_plot_data())
    assert figure.axes[0].collections[0].cmap.name == "magma"
    assert figure.axes[1].collections[0].cmap.name == "viridis"
    plotting.close_without_export(figure)


def test_model_comparison_uses_one_shared_legend_clear_of_axes():
    figure = plotting.plot_model_comparison(synthetic_plot_data())

    assert len(figure.axes) == 2
    assert len(figure.legends) == 1
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert figure.subplotpars.top <= 0.72
    plotting.close_without_export(figure)


def test_property_coupling_uses_two_nonredundant_panels():
    figure = plotting.plot_property_coupling(synthetic_plot_data())

    assert len(figure.axes) == 2
    assert "体积热容" in figure.axes[0].get_ylabel()
    assert "扩散系数" in figure.axes[1].get_ylabel()
    assert "⁻" not in " ".join(axis.get_ylabel() for axis in figure.axes)
    assert len(figure.legends) == 1
    plotting.close_without_export(figure)


def test_main_response_uses_center_average_surface_once():
    figure = plotting.plot_main_response(synthetic_plot_data())

    assert len(figure.axes) == 2
    assert all(len(axis.lines) == 3 for axis in figure.axes)
    assert len(figure.legends) == 1
    assert all(axis.get_legend() is None for axis in figure.axes)
    plotting.close_without_export(figure)


def test_validation_figure_is_compact_and_uses_reported_metrics():
    assert plotting.VALIDATION_METRICS["grid_temperature"][0] == pytest.approx(1.16e-4)
    assert plotting.VALIDATION_METRICS["time_moisture"][1] == pytest.approx(5.58e-6)

    figure = plotting.plot_numerical_validation(synthetic_plot_data())
    assert len(figure.axes) == 2
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    plotting.close_without_export(figure)


def test_all_figures_have_centered_bold_titles_without_panel_letters():
    data = synthetic_plot_data()
    builders = (
        plotting.plot_model_comparison,
        plotting.plot_property_coupling,
        plotting.plot_main_response,
        plotting.plot_field_evolution,
        plotting.plot_numerical_validation,
    )
    for builder in builders:
        figure = builder(data)
        assert figure._suptitle is not None
        assert figure._suptitle.get_position()[0] == pytest.approx(0.5)
        assert figure._suptitle.get_fontweight() == "bold"
        assert figure._suptitle.get_fontsize() >= 12
        axis_text = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert not ({"a", "b", "c", "d"} & set(axis_text))
        plotting.close_without_export(figure)


def test_save_publication_figure_exports_only_png(monkeypatch):
    output_dir = plotting.FIGURE_DIR / "_test_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plotting, "FIGURE_DIR", output_dir)
    figure, axis = plotting.plt.subplots(figsize=(3.5, 2.5))
    axis.plot([0.0, 1.0], [0.0, 1.0])

    try:
        plotting.save_publication_figure(figure, "test_export")
        assert [path.name for path in output_dir.iterdir()] == ["test_export.png"]
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
