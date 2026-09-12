import numpy as np
import pytest
import shutil

import 问题一_绘图 as plotting


def test_prepend_initial_state_adds_t_zero_without_mutating_field():
    times = np.array([1.0, 2.0])
    field = np.array([[1.5, 1.0], [1.4, 0.9]])
    new_times, new_field = plotting.prepend_initial_state(times, field, 2.0)

    np.testing.assert_allclose(new_times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(new_field[0], [2.0, 2.0])
    np.testing.assert_allclose(new_field[1:], field)
    assert not np.shares_memory(new_field, field)


def test_max_radial_error_reduces_only_over_radial_axis():
    reference = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    refined = np.array([[1.1, 1.8, 3.0], [1.5, 2.2, 1.9]])

    np.testing.assert_allclose(
        plotting.max_radial_error(reference, refined), [0.2, 0.5]
    )


def test_validate_comparable_outputs_rejects_mismatched_grids():
    with pytest.raises(ValueError, match="时间轴"):
        plotting.validate_comparable_outputs(
            np.array([1.0, 2.0]), np.array([1.0, 2.5]),
            np.array([0.0, 1.0]), np.array([0.0, 1.0]),
        )


def test_new_figure_contract_contains_exactly_five_outputs():
    assert len(plotting.NEW_FIGURE_BASES) == 5
    assert len(set(plotting.NEW_FIGURE_BASES)) == 5
    assert plotting.FIGURE_FORMATS == ("png",)


def test_load_plot_data_matches_official_result_and_has_refinement_fields():
    data = plotting.load_plot_data()
    assert data["times_s"].shape == data["temperature_field"].shape[:1]
    assert data["temperature_field"].shape == data["moisture_field"].shape
    assert data["temperature_field"].shape[1] == len(data["radii_cm"])
    assert data["fine_space_temperature"].shape == data["temperature_field"].shape
    assert data["fine_time_moisture"].shape == data["moisture_field"].shape
    assert np.all(np.diff(data["times_s"]) > 0.0)


def test_save_publication_figure_exports_only_png(monkeypatch):
    output_dir = plotting.FIGURE_DIR / "_test_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plotting, "FIGURE_DIR", output_dir)
    figure, axis = plotting.plt.subplots(figsize=(3.5, 2.5))
    axis.plot([0.0, 1.0], [0.0, 1.0])

    try:
        plotting.save_publication_figure(figure, "test_export")

        output = output_dir / "test_export.png"
        assert output.exists()
        assert output.stat().st_size > 0
        assert [path.name for path in output_dir.iterdir()] == ["test_export.png"]
        for suffix in (".pdf", ".svg", ".tiff"):
            assert not (output_dir / f"test_export{suffix}").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_boundary_figure_has_two_comparable_axes():
    data = plotting.load_plot_data()
    figure = plotting.plot_boundary_conditions(data)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xlabel() == figure.axes[1].get_xlabel()
    assert all(axis.get_xlim()[1] == pytest.approx(0.5) for axis in figure.axes)
    all_text = " ".join(
        [text.get_text() for text in figure.texts]
        + [text.get_text() for axis in figure.axes for text in axis.texts]
    )
    assert "求解窗口" not in all_text
    assert "稳定段" not in all_text
    assert "采样点" not in all_text
    plotting.close_without_export(figure)


def test_main_response_figure_keeps_only_two_nonredundant_panels():
    data = plotting.load_plot_data()
    figure = plotting.plot_main_response(data)

    assert len(figure.axes) == 2
    assert [axis.get_xlabel() for axis in figure.axes] == ["时间 / h", "时间 / h"]
    assert all(
        text.get_bbox_patch() is None
        for axis in figure.axes
        for text in axis.texts
    )
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert len(figure.legends) == 1
    plotting.close_without_export(figure)


def test_style_prefers_microsoft_yahei_and_disables_gridlines():
    assert plotting.CJK_FONT_CANDIDATES[0] == "Microsoft YaHei"
    assert plotting.mpl.rcParams["font.sans-serif"][0] == "Microsoft YaHei"
    figure, axis = plotting.plt.subplots()
    plotting._style_axis(axis)
    assert not any(line.get_visible() for line in axis.get_ygridlines())
    plotting.close_without_export(figure)


def test_reference_palette_is_used_consistently():
    assert plotting.REFERENCE_PALETTE == (
        "#44757A", "#452A3D", "#D44C3C", "#EED5B7"
    )
    data = plotting.load_plot_data()

    response = plotting.plot_main_response(data)
    assert [line.get_color() for line in response.axes[0].lines] == [
        "#44757A", "#452A3D", "#D44C3C"
    ]
    plotting.close_without_export(response)

    fields = plotting.plot_field_evolution(data)
    assert fields.axes[0].collections[0].cmap.name == "magma"
    assert fields.axes[1].collections[0].cmap.name == "viridis"
    plotting.close_without_export(fields)

    profiles = plotting.plot_radial_profiles(data)
    assert [line.get_color() for line in profiles.axes[0].lines] == [
        "#44757A", "#452A3D", "#B88E63", "#D44C3C"
    ]
    plotting.close_without_export(profiles)


def test_figure_labels_avoid_unsupported_minus_glyph():
    data = plotting.load_plot_data()
    builders = (
        plotting.plot_boundary_conditions,
        plotting.plot_main_response,
        plotting.plot_field_evolution,
        plotting.plot_radial_profiles,
        plotting.plot_numerical_validation,
    )
    for builder in builders:
        figure = builder(data)
        labels = []
        for axis in figure.axes:
            labels.extend(
                [axis.get_title(), axis.get_xlabel(), axis.get_ylabel()]
                + [text.get_text() for text in axis.texts]
            )
        assert "−" not in " ".join(labels)
        plotting.close_without_export(figure)


def test_figures_use_top_titles_without_panel_letters():
    data = plotting.load_plot_data()
    builders = (
        plotting.plot_boundary_conditions,
        plotting.plot_main_response,
        plotting.plot_field_evolution,
        plotting.plot_radial_profiles,
        plotting.plot_numerical_validation,
    )
    for builder in builders:
        figure = builder(data)
        assert figure._suptitle is not None
        assert figure._suptitle.get_text().strip()
        assert figure._suptitle.get_fontweight() == "bold"
        assert figure._suptitle.get_fontsize() >= 12
        axis_text = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert not ({"a", "b", "c", "d"} & set(axis_text))
        plotting.close_without_export(figure)


def test_field_evolution_uses_explicit_initial_state():
    data = plotting.load_plot_data()
    figure = plotting.plot_field_evolution(data)

    assert figure.axes[0].get_xlim()[0] <= 0.0
    assert figure.axes[1].get_xlim()[0] <= 0.0
    plotting.close_without_export(figure)


def test_radial_profiles_use_the_four_declared_snapshot_times():
    data = plotting.load_plot_data()
    figure = plotting.plot_radial_profiles(data)

    assert len(figure.axes) == 2
    assert all(len(axis.lines) == 4 for axis in figure.axes)
    plotting.close_without_export(figure)


def test_validation_errors_match_reported_problem_one_scales():
    data = plotting.load_plot_data()
    errors = plotting.build_validation_errors(data)

    assert set(errors) == {
        "grid_temperature", "grid_moisture",
        "time_temperature", "time_moisture",
    }
    assert np.max(errors["grid_temperature"]) == pytest.approx(2.597288e-5, rel=5e-3)
    assert np.max(errors["grid_moisture"]) == pytest.approx(6.840443e-3, rel=5e-3)
    assert np.max(errors["time_temperature"]) == pytest.approx(9.547366e-4, rel=5e-3)
    assert np.max(errors["time_moisture"]) == pytest.approx(7.649922e-4, rel=5e-3)


def test_numerical_validation_figure_has_four_panels():
    data = plotting.load_plot_data()
    figure = plotting.plot_numerical_validation(data)

    assert len(figure.axes) == 4
    assert not any("基准诊断" in text.get_text() for text in figure.texts)
    assert all(
        text.get_bbox_patch() is None
        for axis in figure.axes
        for text in axis.texts
    )
    figure.canvas.draw()
    tick_text = " ".join(
        tick.get_text() for axis in figure.axes for tick in axis.get_yticklabels()
    )
    assert "−" not in tick_text
    assert "$" not in tick_text
    plotting.close_without_export(figure)


def test_export_all_builds_five_named_figures(monkeypatch):
    data = plotting.load_plot_data()
    saved = []
    output_dir = plotting.FIGURE_DIR / "_test_export_all"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plotting, "FIGURE_DIR", output_dir)

    def fake_save(figure, base_name, exclude_axes=None):
        saved.append(base_name)
        plotting.close_without_export(figure)
        for suffix in plotting.FIGURE_FORMATS:
            (output_dir / f"{base_name}.{suffix}").write_bytes(b"test")

    monkeypatch.setattr(plotting, "save_publication_figure", fake_save)
    monkeypatch.setattr(plotting, "remove_legacy_problem_one_outputs", lambda: None)

    try:
        plotting.export_all(data)
        assert saved == list(plotting.NEW_FIGURE_BASES)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
