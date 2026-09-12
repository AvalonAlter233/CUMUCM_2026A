from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result1.xlsx"

PELLET_RADIUS = 0.02
PELLET_DENSITY = 820.0
PELLET_HEAT_CAPACITY = 2600.0
THERMAL_CONDUCTIVITY = 0.36
CONVECTIVE_HEAT_COEFF = 25.0
CONVECTIVE_MASS_COEFF = 8.0e-7

INITIAL_TEMPERATURE = 28.0
INITIAL_MOISTURE = 2.55

END_TIME = 1800.0
TIME_STEP = 1.0
INTERNAL_CELL_COUNT = 160
OUTPUT_TIME_INTERVAL = 1.0
OUTPUT_RADII = np.linspace(0.0, PELLET_RADIUS, 21)

CONVERGENCE_TOL = 1.0e-10
MAX_ITERATIONS = 30


class DryingRoomBoundary(NamedTuple):

    times: np.ndarray
    temperatures: np.ndarray
    moisture: np.ndarray


class RadialGrid(NamedTuple):

    centers: np.ndarray
    faces: np.ndarray
    volume_factors: np.ndarray
    spacing: float


class SimulationDetails(NamedTuple):

    times: np.ndarray
    output_radii: np.ndarray
    temperature_field: np.ndarray
    moisture_field: np.ndarray
    average_temperature: np.ndarray
    average_moisture: np.ndarray
    diagnostics: dict[str, float]


def read_drying_boundary(path: Path) -> DryingRoomBoundary:
    book = load_workbook(path, data_only=True, read_only=True)
    sheet = book.active
    rows = [
        row for row in sheet.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    data = np.asarray(rows, dtype=float)

    if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] < 2:
        raise ValueError("附件一必须包含时间、温度、水分浓度三列有效数据。")
    if not np.all(np.isfinite(data[:, :3])):
        raise ValueError("附件一的前三列存在空值或非有限数值。")

    times = data[:, 0]
    if not np.all(np.diff(times) > 0):
        raise ValueError("附件一中的时间必须严格递增。")

    return DryingRoomBoundary(
        times=times,
        temperatures=data[:, 1],
        moisture=data[:, 2],
    )


def linear_interp(t: float, times: np.ndarray, values: np.ndarray) -> float:
    if t < times[0] or t > times[-1]:
        raise ValueError(
            f"计算时刻 {t:g} s 超出附件一范围 "
            f"[{times[0]:g}, {times[-1]:g}] s。"
        )
    return float(np.interp(t, times, values))


def moisture_diffusivity(moisture: np.ndarray) -> np.ndarray:
    c_safe = np.maximum(moisture, 1.0e-12)
    return 7.0e-9 * np.exp(-0.89 / c_safe)


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.maximum(left + right, 1.0e-30)
    return 2.0 * left * right / denom


def build_radial_grid(cell_count: int) -> RadialGrid:
    if cell_count < 2:
        raise ValueError("径向有限体积单元数至少为 2。")

    faces = np.linspace(0.0, PELLET_RADIUS, cell_count + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    vol = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    return RadialGrid(
        centers=centers,
        faces=faces,
        volume_factors=vol,
        spacing=float(faces[1] - faces[0]),
    )


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    diagonal = diagonal.astype(float, copy=True)
    right_hand_side = right_hand_side.astype(float, copy=True)
    upper = upper.astype(float, copy=True)

    for i in range(1, len(diagonal)):
        if abs(diagonal[i - 1]) < 1.0e-30:
            raise FloatingPointError("三对角方程组出现近零主对角元。")
        factor = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        right_hand_side[i] -= factor * right_hand_side[i - 1]

    sol = np.empty_like(right_hand_side)
    sol[-1] = right_hand_side[-1] / diagonal[-1]
    for i in range(len(diagonal) - 2, -1, -1):
        sol[i] = (
            right_hand_side[i] - upper[i] * sol[i + 1]
        ) / diagonal[i]
    return sol


def effective_surface_transfer(
    internal_coefficient: float,
    boundary_coefficient: float,
    half_cell_width: float,
) -> float:
    if internal_coefficient <= 0.0 or boundary_coefficient <= 0.0:
        raise ValueError("内部传输系数和表面传递系数必须为正。")
    return 1.0 / (
        half_cell_width / internal_coefficient + 1.0 / boundary_coefficient
    )


def assemble_control_volume_system(
    storage_coefficients: np.ndarray,
    old_values: np.ndarray,
    grid: RadialGrid,
    interface_coefficients: np.ndarray,
    surface_internal_coefficient: float,
    surface_transfer_coefficient: float,
    boundary_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_count = len(grid.centers)
    lower = np.zeros(cell_count - 1)
    upper = np.zeros(cell_count - 1)
    diag = storage_coefficients.copy()
    rhs = storage_coefficients * old_values

    g_face = grid.faces[1:-1] * interface_coefficients / grid.spacing
    diag[:-1] += g_face
    diag[1:] += g_face
    upper[:] = -g_face
    lower[:] = -g_face

    h_eff = effective_surface_transfer(
        internal_coefficient=surface_internal_coefficient,
        boundary_coefficient=surface_transfer_coefficient,
        half_cell_width=0.5 * grid.spacing,
    )
    g_surf = PELLET_RADIUS * h_eff
    diag[-1] += g_surf
    rhs[-1] += g_surf * boundary_value
    return lower, diag, upper, rhs


def surface_value_from_last_cell(
    last_cell_value: float,
    internal_coefficient: float,
    boundary_coefficient: float,
    boundary_value: float,
    grid_spacing: float,
) -> float:
    g_inner = 2.0 * internal_coefficient / grid_spacing
    return float(
        (g_inner * last_cell_value + boundary_coefficient * boundary_value)
        / (g_inner + boundary_coefficient)
    )


def reconstruct_output_profile(
    cell_values: np.ndarray,
    grid: RadialGrid,
    output_radii: np.ndarray,
    surface_internal_coefficient: float,
    surface_transfer_coefficient: float,
    boundary_value: float,
) -> np.ndarray:
    r1_sq = grid.centers[0] ** 2
    r2_sq = grid.centers[1] ** 2
    axis_val = (
        r2_sq * cell_values[0]
        - r1_sq * cell_values[1]
    ) / (r2_sq - r1_sq)

    val_surf = surface_value_from_last_cell(
        last_cell_value=float(cell_values[-1]),
        internal_coefficient=surface_internal_coefficient,
        boundary_coefficient=surface_transfer_coefficient,
        boundary_value=boundary_value,
        grid_spacing=grid.spacing,
    )

    r_fit = np.concatenate(([0.0], grid.centers, [PELLET_RADIUS]))
    val_fit = np.concatenate(([axis_val], cell_values, [val_surf]))
    return np.interp(output_radii, r_fit, val_fit)


def radial_volume_average(cell_values: np.ndarray, grid: RadialGrid) -> float:
    return float(2.0 * np.sum(grid.volume_factors * cell_values) / PELLET_RADIUS**2)


def relative_balance_residual(storage_rate: float, outward_flux: float) -> float:
    scale = abs(storage_rate) + abs(outward_flux) + 1.0e-30
    return abs(storage_rate + outward_flux) / scale


def advance_temperature(
    old_temperature: np.ndarray,
    current_time: float,
    time_step: float,
    grid: RadialGrid,
    boundary: DryingRoomBoundary,
    heat_transfer_coefficient: float,
) -> np.ndarray:
    storage = (
        PELLET_DENSITY * PELLET_HEAT_CAPACITY * grid.volume_factors / time_step
    )
    k_face = np.full(len(grid.centers) - 1, THERMAL_CONDUCTIVITY)
    temp_air = linear_interp(current_time, boundary.times, boundary.temperatures)

    lin_sys = assemble_control_volume_system(
        storage_coefficients=storage,
        old_values=old_temperature,
        grid=grid,
        interface_coefficients=k_face,
        surface_internal_coefficient=THERMAL_CONDUCTIVITY,
        surface_transfer_coefficient=heat_transfer_coefficient,
        boundary_value=temp_air,
    )
    return solve_tridiagonal(*lin_sys)


def advance_moisture(
    old_moisture: np.ndarray,
    current_time: float,
    time_step: float,
    grid: RadialGrid,
    boundary: DryingRoomBoundary,
    mass_transfer_coefficient: float,
) -> tuple[np.ndarray, int, float]:
    storage = grid.volume_factors / time_step
    water_iter = old_moisture.copy()
    water_air = linear_interp(current_time, boundary.times, boundary.moisture)

    for iteration in range(1, MAX_ITERATIONS + 1):
        d_cell = moisture_diffusivity(water_iter)
        d_face = harmonic_mean(d_cell[:-1], d_cell[1:])
        lin_sys = assemble_control_volume_system(
            storage_coefficients=storage,
            old_values=old_moisture,
            grid=grid,
            interface_coefficients=d_face,
            surface_internal_coefficient=float(d_cell[-1]),
            surface_transfer_coefficient=mass_transfer_coefficient,
            boundary_value=water_air,
        )
        water_new = solve_tridiagonal(*lin_sys)
        final_error = float(
            np.max(
                np.abs(water_new - water_iter)
                / (1.0 + np.abs(water_new))
            )
        )
        water_iter = water_new
        if final_error < CONVERGENCE_TOL:
            return water_new, iteration, final_error

    raise RuntimeError(
        f"{current_time:g} s 的含水率 Picard 迭代未收敛，"
        f"最终相对变化量为 {final_error:.3e}。"
    )


def simulate_problem_one(
    end_time: float = END_TIME,
    time_step: float = TIME_STEP,
    cell_count: int = INTERNAL_CELL_COUNT,
    output_interval: float = OUTPUT_TIME_INTERVAL,
    heat_transfer_coefficient: float = CONVECTIVE_HEAT_COEFF,
    mass_transfer_coefficient: float = CONVECTIVE_MASS_COEFF,
) -> SimulationDetails:
    if min(end_time, time_step, output_interval) <= 0.0:
        raise ValueError("终止时间、时间步长和输出间隔必须为正。")
    n_steps = int(round(end_time / time_step))
    save_every = int(round(output_interval / time_step))
    if not np.isclose(n_steps * time_step, end_time):
        raise ValueError("终止时间必须是时间步长的整数倍。")
    if save_every < 1 or not np.isclose(save_every * time_step, output_interval):
        raise ValueError("输出间隔必须是时间步长的整数倍。")

    boundary = read_drying_boundary(INPUT_FILE)
    if end_time > boundary.times[-1]:
        raise ValueError("附件一边界数据没有覆盖要求的计算时段。")
    mesh = build_radial_grid(cell_count)

    temp = np.full(cell_count, INITIAL_TEMPERATURE, dtype=float)
    water = np.full(cell_count, INITIAL_MOISTURE, dtype=float)

    time_log: list[float] = []
    temp_log: list[np.ndarray] = []
    water_log: list[np.ndarray] = []
    temp_mean_log: list[float] = []
    water_mean_log: list[float] = []
    iter_log: list[int] = []
    max_iter_err = 0.0
    max_heat_err = 0.0
    max_water_err = 0.0

    for step in range(1, n_steps + 1):
        t_now = step * time_step
        temp_old = temp.copy()
        water_old = water.copy()

        temp = advance_temperature(
            temp_old,
            t_now,
            time_step,
            mesh,
            boundary,
            heat_transfer_coefficient,
        )
        water, n_iter, iter_err = advance_moisture(
            water_old,
            t_now,
            time_step,
            mesh,
            boundary,
            mass_transfer_coefficient,
        )

        if not np.all(np.isfinite(temp)) or not np.all(np.isfinite(water)):
            raise FloatingPointError(f"{t_now:g} s 出现非有限场变量。")
        if np.min(water) <= 0.0:
            raise FloatingPointError(f"{t_now:g} s 出现非正含水率。")

        temp_air = linear_interp(t_now, boundary.times, boundary.temperatures)
        water_air = linear_interp(t_now, boundary.times, boundary.moisture)
        temp_surf = surface_value_from_last_cell(
            float(temp[-1]),
            THERMAL_CONDUCTIVITY,
            heat_transfer_coefficient,
            temp_air,
            mesh.spacing,
        )
        d_surf = float(moisture_diffusivity(water[-1:])[0])
        water_surf = surface_value_from_last_cell(
            float(water[-1]),
            d_surf,
            mass_transfer_coefficient,
            water_air,
            mesh.spacing,
        )

        heat_rate = float(
            np.sum(
                PELLET_DENSITY * PELLET_HEAT_CAPACITY * mesh.volume_factors
                * (temp - temp_old) / time_step
            )
        )
        water_rate = float(
            np.sum(mesh.volume_factors * (water - water_old) / time_step)
        )
        heat_out = (
            PELLET_RADIUS * heat_transfer_coefficient
            * (temp_surf - temp_air)
        )
        water_out = (
            PELLET_RADIUS * mass_transfer_coefficient
            * (water_surf - water_air)
        )
        max_heat_err = max(
            max_heat_err,
            relative_balance_residual(heat_rate, heat_out),
        )
        max_water_err = max(
            max_water_err,
            relative_balance_residual(water_rate, water_out),
        )
        iter_log.append(n_iter)
        max_iter_err = max(max_iter_err, iter_err)

        if step % save_every == 0:
            time_log.append(t_now)
            temp_log.append(
                reconstruct_output_profile(
                    temp,
                    mesh,
                    OUTPUT_RADII,
                    THERMAL_CONDUCTIVITY,
                    heat_transfer_coefficient,
                    temp_air,
                )
            )
            water_log.append(
                reconstruct_output_profile(
                    water,
                    mesh,
                    OUTPUT_RADII,
                    d_surf,
                    mass_transfer_coefficient,
                    water_air,
                )
            )
            temp_mean_log.append(radial_volume_average(temp, mesh))
            water_mean_log.append(radial_volume_average(water, mesh))

    diagnostics = {
        "internal_cell_count": float(cell_count),
        "time_step": float(time_step),
        "max_picard_iterations": float(max(iter_log)),
        "mean_picard_iterations": float(np.mean(iter_log)),
        "max_final_picard_error": float(max_iter_err),
        "max_heat_balance_residual": float(max_heat_err),
        "max_moisture_balance_residual": float(max_water_err),
    }
    return SimulationDetails(
        times=np.asarray(time_log),
        output_radii=OUTPUT_RADII.copy(),
        temperature_field=np.asarray(temp_log),
        moisture_field=np.asarray(water_log),
        average_temperature=np.asarray(temp_mean_log),
        average_moisture=np.asarray(water_mean_log),
        diagnostics=diagnostics,
    )


def solve_problem_one() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    result = simulate_problem_one()
    return result.times, result.output_radii, result.temperature_field, result.moisture_field


def write_result_workbook(
    times: np.ndarray,
    output_radii: np.ndarray,
    temperature_field: np.ndarray,
    moisture_field: np.ndarray,
) -> None:
    book = load_workbook(OUTPUT_FILE)
    if len(book.worksheets) < 2:
        raise ValueError("result1.xlsx 必须包含温度和水分浓度两个工作表。")

    for sheet, field in zip(
        book.worksheets[:2], (temperature_field, moisture_field)
    ):
        if sheet.max_row > 1:
            sheet.delete_rows(2, sheet.max_row - 1)

        sheet.cell(1, 1).value = "时间\\到药材中心的距离"
        sheet.cell(1, 1).number_format = "@"
        for column, radius in enumerate(output_radii, start=2):
            sheet.cell(1, column).value = round(float(radius * 100.0), 1)
            sheet.cell(1, column).number_format = "0.0"

        for row, t_now in enumerate(times, start=2):
            sheet.cell(row, 1).value = int(round(float(t_now)))
            sheet.cell(row, 1).number_format = "0"
            for column, value in enumerate(field[row - 2], start=2):
                sheet.cell(row, column).value = round(float(value), 4)
                sheet.cell(row, column).number_format = "0.0000"

    book.save(OUTPUT_FILE)


def report_largest_difference(
    label: str,
    times: np.ndarray,
    radii: np.ndarray,
    first_field: np.ndarray,
    second_field: np.ndarray,
) -> None:
    diff = np.abs(first_field - second_field)
    it, ir = np.unravel_index(
        int(np.argmax(diff)), diff.shape
    )
    print(
        f"{label}：最大绝对差={diff[it, ir]:.6e}，"
        f"位置为 t={times[it]:g} s、"
        f"r={radii[ir] * 100:g} cm；"
        f"终点最大差={np.max(diff[-1]):.6e}。"
    )


def run_numerical_validation(reference: SimulationDetails) -> None:
    space_fine = simulate_problem_one(cell_count=320)
    time_fine = simulate_problem_one(cell_count=160, time_step=0.5)

    print("问题一数值检验：")
    report_largest_difference(
        "N=160 与 N=320 的温度场",
        reference.times,
        reference.output_radii,
        reference.temperature_field,
        space_fine.temperature_field,
    )
    report_largest_difference(
        "N=160 与 N=320 的含水率场",
        reference.times,
        reference.output_radii,
        reference.moisture_field,
        space_fine.moisture_field,
    )
    report_largest_difference(
        "时间步 1 s 与 0.5 s 的温度场",
        reference.times,
        reference.output_radii,
        reference.temperature_field,
        time_fine.temperature_field,
    )
    report_largest_difference(
        "时间步 1 s 与 0.5 s 的含水率场",
        reference.times,
        reference.output_radii,
        reference.moisture_field,
        time_fine.moisture_field,
    )
    print(f"主计算诊断：{reference.diagnostics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="求解问题一的一维径向传热传质模型")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="额外运行空间和时间步长检验，只在终端输出诊断",
    )
    args = parser.parse_args()

    result = simulate_problem_one()
    write_result_workbook(
        result.times,
        result.output_radii,
        result.temperature_field,
        result.moisture_field,
    )
    print(f"问题一计算完成，结果已写入：{OUTPUT_FILE}")
    if args.validate:
        run_numerical_validation(result)


if __name__ == "__main__":
    main()
