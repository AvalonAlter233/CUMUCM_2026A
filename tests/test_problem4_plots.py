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


def test_plotting_main_generates_seven_nonempty_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "FIGURE_DIR", Path(tmp_path))
    plotting.main()
    figures = sorted(Path(tmp_path).glob("*.png"))
    assert len(figures) == 7
    assert all(path.stat().st_size > 10_000 for path in figures)
