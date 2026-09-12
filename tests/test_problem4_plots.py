from pathlib import Path

import numpy as np
import pytest

import 问题四_绘图 as plotting


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


def test_plotting_main_generates_seven_nonempty_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "FIGURE_DIR", Path(tmp_path))
    plotting.main()
    figures = sorted(Path(tmp_path).glob("*.png"))
    assert len(figures) == 7
    assert all(path.stat().st_size > 10_000 for path in figures)
