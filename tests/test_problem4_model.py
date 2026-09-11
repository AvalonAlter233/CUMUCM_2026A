import json
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

import 问题3_求解 as problem3
import 问题4_求解 as problem4


def test_radius_history_is_monotone_and_plateaus_after_measurements():
    history = problem4.read_radius_history(problem4.RADIUS_FILE)
    assert np.all(np.diff(history.times) > 0.0)
    assert np.all(np.diff(history.radii) <= 1.0e-12)
    assert history.radii[0] == pytest.approx(0.02000)
    assert history.radii[-1] == pytest.approx(0.01198)
    assert problem4.radius_value(history.times[-1] + 3600.0, history) == pytest.approx(
        history.radii[-1]
    )


def test_appendix_four_property_formulas_match_specification():
    moisture = np.array([0.15, 1.0, 2.55])
    temperature = np.array([28.0, 40.0, 50.0])
    assert np.allclose(problem4.density(moisture), 760.0 + 90.0 * moisture)
    assert np.allclose(
        problem4.heat_capacity(moisture),
        1850.0 + 2150.0 * moisture / (moisture + 1.0),
    )
    assert np.allclose(
        problem4.thermal_conductivity(moisture),
        0.12 + 0.20 * moisture / (moisture + 1.0),
    )
    expected_d = (
        4.2e-4
        * np.exp(-0.30 / moisture)
        * np.exp(-3850.0 / (temperature + 273.15))
    )
    assert np.allclose(problem4.moisture_diffusivity(moisture, temperature), expected_d)


def test_reference_grid_has_correct_axisymmetric_volume():
    grid = problem4.build_reference_grid(320)
    assert len(grid.centers) == 320
    assert grid.spacing == pytest.approx(1.0 / 320.0)
    assert np.all(np.diff(grid.faces) > 0.0)
    assert np.sum(grid.volumes) == pytest.approx(0.5)


def test_reference_grid_rejects_too_few_cells():
    with pytest.raises(ValueError, match="至少为 4"):
        problem4.build_reference_grid(3)


def test_reference_system_has_positive_diagonal_and_expected_shape():
    grid = problem4.build_reference_grid(8)
    old = np.full(8, 1.0)
    storage = np.ones(8)
    transport = np.full(8, 2.0e-9)
    lower, diagonal, upper, rhs = problem4.assemble_reference_system(
        old, storage, transport, 8.0e-7, 0.05, grid, 0.02, 30.0
    )
    assert lower.shape == upper.shape == (7,)
    assert diagonal.shape == rhs.shape == (8,)
    assert np.all(diagonal > 0.0)
    assert np.all(lower < 0.0)
    assert np.all(upper < 0.0)


def test_tridiagonal_solver_matches_dense_solution():
    lower = np.array([-1.0, -1.0])
    diagonal = np.array([4.0, 4.0, 4.0])
    upper = np.array([-1.0, -1.0])
    rhs = np.array([2.0, 4.0, 10.0])
    dense = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
    assert np.allclose(
        problem4.solve_tridiagonal(lower, diagonal, upper, rhs),
        np.linalg.solve(dense, rhs),
    )


def test_reconstruction_returns_fixed_nodes_and_moving_surface():
    grid = problem4.build_reference_grid(8)
    values = 1.0 - 0.5 * grid.centers
    transport = np.full(8, 2.0e-9)
    nodes = np.array([0.0, 0.005, 0.010])
    result = problem4.reconstruct_output_field(
        values, transport, 8.0e-7, 0.05, nodes, grid, 0.02
    )
    assert result.shape == (4,)
    assert np.all(np.diff(result) <= 0.0)
    assert 0.05 < result[-1] < values[-1]


def test_reconstruction_rejects_nodes_outside_current_radius():
    grid = problem4.build_reference_grid(8)
    with pytest.raises(ValueError, match="药材内部"):
        problem4.reconstruct_output_field(
            np.ones(8),
            np.ones(8),
            8.0e-7,
            0.05,
            np.array([0.0, 0.021]),
            grid,
            0.02,
        )


def test_template_nodes_and_headers_follow_result4_layout():
    nodes = problem4.template_output_nodes()
    assert np.allclose(nodes, np.arange(0.0, 0.012, 0.001))
    assert problem4.output_headers(nodes) == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        "药材表面",
    ]


def test_result_workbook_and_diagnostics_confirm_official_end_time():
    workbook = load_workbook(problem4.OUTPUT_FILE, read_only=True, data_only=True)
    sheet = workbook.active
    previous = np.array(
        [sheet.cell(sheet.max_row - 1, c).value for c in range(2, sheet.max_column + 1)]
    )
    final = np.array(
        [sheet.cell(sheet.max_row, c).value for c in range(2, sheet.max_column + 1)]
    )
    assert sheet.cell(sheet.max_row, 1).value == 184020
    assert np.max(previous) >= problem4.CRITICAL_MOISTURE
    assert np.max(final) < problem4.CRITICAL_MOISTURE

    diagnostics_path = Path(problem4.DIAGNOSTICS_FILE)
    report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    baseline = report["baseline"]
    assert baseline["discrete_threshold_time_s"] == pytest.approx(184020.0)
    assert baseline["continuous_threshold_time_h"] == pytest.approx(51.1151668549404)
    assert baseline["center_controls_threshold"]
    assert baseline["all_domain_checked_every_step"]
    assert baseline["radially_nonincreasing_every_step"]
    assert baseline["moisture_balance_relative_imbalance"] < 1.0e-5

