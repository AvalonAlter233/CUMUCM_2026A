from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result2.xlsx"

PELLET_RADIUS = 0.02
CONVECTIVE_HEAT_COEFF = 25.0
CONVECTIVE_MASS_COEFF = 8.0e-7
INITIAL_TEMPERATURE = 28.0
INITIAL_MOISTURE = 2.55

END_TIME = 10800.0
TIME_STEP = 1.0
INTERNAL_CELL_COUNT = 160
OUTPUT_TIME_INTERVAL = 1.0
OUTPUT_RADII = np.linspace(0.0, PELLET_RADIUS, 21)

CONVERGENCE_TOL = 1.0e-9
MAX_ITERATIONS = 60
RELAXATION_FACTOR = 0.8


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
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("附件一中的时间必须严格递增。")

    return DryingRoomBoundary(
        times=data[:, 0],
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


def density(moisture: np.ndarray) -> np.ndarray:
    return 650.0 + 128.0 * moisture


def heat_capacity(moisture: np.ndarray) -> np.ndarray:
    return 1450.0 + 2736.0 * moisture / (moisture + 1.0)


def thermal_conductivity(moisture: np.ndarray) -> np.ndarray:
    return 0.21 + 0.38 * moisture / (moisture + 1.0)


def moisture_diffusivity(
    moisture: np.ndarray,
    temperature_celsius: np.ndarray,
) -> np.ndarray:
    c_safe = np.maximum(moisture, 1.0e-12)
    temp_k = temperature_celsius + 273.15
    return (
        2.4e-3
        * np.exp(-0.45 / c_safe)
        * np.exp(-3850.0 / temp_k)
    )


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
        float(cell_values[-1]),
        surface_internal_coefficient,
        surface_transfer_coefficient,
        boundary_value,
        grid.spacing,
    )
    r_fit = np.concatenate(([0.0], grid.centers, [PELLET_RADIUS]))
    val_fit = np.concatenate(([axis_val], cell_values, [val_surf]))
    return np.interp(output_radii, r_fit, val_fit)


def radial_volume_average(cell_values: np.ndarray, grid: RadialGrid) -> float:
    return float(2.0 * np.sum(grid.volume_factors * cell_values) / PELLET_RADIUS**2)


def relative_balance_residual(storage_rate: float, outward_flux: float) -> float:
    scale = abs(storage_rate) + abs(outward_flux) + 1.0e-30
    return abs(storage_rate + outward_flux) / scale


def solve_temperature_candidate(
    old_temperature: np.ndarray,
    reference_moisture: np.ndarray,
    current_time: float,
    time_step: float,
    grid: RadialGrid,
    boundary: DryingRoomBoundary,
    heat_transfer_coefficient: float,
) -> np.ndarray:
    rho = density(reference_moisture)
    cp = heat_capacity(reference_moisture)
    k_cell = thermal_conductivity(reference_moisture)
    storage = rho * cp * grid.volume_factors / time_step
    k_face = harmonic_mean(k_cell[:-1], k_cell[1:])
    temp_air = linear_interp(current_time, boundary.times, boundary.temperatures)

    lin_sys = assemble_control_volume_system(
        storage,
        old_temperature,
        grid,
        k_face,
        float(k_cell[-1]),
        heat_transfer_coefficient,
        temp_air,
    )
    return solve_tridiagonal(*lin_sys)


def solve_moisture_candidate(
    old_moisture: np.ndarray,
    reference_temperature: np.ndarray,
    reference_moisture: np.ndarray,
    current_time: float,
    time_step: float,
    grid: RadialGrid,
    boundary: DryingRoomBoundary,
    mass_transfer_coefficient: float,
) -> np.ndarray:
    d_cell = moisture_diffusivity(reference_moisture, reference_temperature)
    d_face = harmonic_mean(d_cell[:-1], d_cell[1:])
    storage = grid.volume_factors / time_step
    water_air = linear_interp(current_time, boundary.times, boundary.moisture)

    lin_sys = assemble_control_volume_system(
        storage,
        old_moisture,
        grid,
        d_face,
        float(d_cell[-1]),
        mass_transfer_coefficient,
        water_air,
    )
    return solve_tridiagonal(*lin_sys)


def simulate_problem_two(
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
    max_temp_err = 0.0
    max_moisture_err = 0.0
    max_heat_err = 0.0
    max_water_err = 0.0
    rho_min = float("inf")
    rho_max = -float("inf")
    cp_min = float("inf")
    cp_max = -float("inf")
    k_min = float("inf")
    k_max = -float("inf")
    d_min = float("inf")
    d_max = -float("inf")

    for step in range(1, n_steps + 1):
        t_now = step * time_step
        temp_old = temp.copy()
        water_old = water.copy()
        temp_iter = temp.copy()
        water_iter = water.copy()

        for iteration in range(1, MAX_ITERATIONS + 1):
            temp_trial = solve_temperature_candidate(
                temp_old,
                water_iter,
                t_now,
                time_step,
                mesh,
                boundary,
                heat_transfer_coefficient,
            )
            water_trial = solve_moisture_candidate(
                water_old,
                temp_trial,
                water_iter,
                t_now,
                time_step,
                mesh,
                boundary,
                mass_transfer_coefficient,
            )
            temp_new = (
                RELAXATION_FACTOR * temp_trial
                + (1.0 - RELAXATION_FACTOR) * temp_iter
            )
            water_new = (
                RELAXATION_FACTOR * water_trial
                + (1.0 - RELAXATION_FACTOR) * water_iter
            )
            err_t = float(
                np.max(
                    np.abs(temp_new - temp_iter)
                    / (1.0 + np.abs(temp_new))
                )
            )
            err_c = float(
                np.max(
                    np.abs(water_new - water_iter)
                    / (1.0 + np.abs(water_new))
                )
            )
            temp_iter = temp_new
            water_iter = water_new
            if max(err_t, err_c) < CONVERGENCE_TOL:
                break
        else:
            raise RuntimeError(
                f"{t_now:g} s 的热湿 Picard 迭代未收敛，"
                f"温度变化量={err_t:.3e}，"
                f"含水率变化量={err_c:.3e}。"
            )

        temp = temp_iter
        water = water_iter
        if not np.all(np.isfinite(temp)) or not np.all(np.isfinite(water)):
            raise FloatingPointError(f"{t_now:g} s 出现非有限场变量。")
        if np.min(water) <= 0.0:
            raise FloatingPointError(f"{t_now:g} s 出现非正含水率。")

        rho = density(water)
        cp = heat_capacity(water)
        k_cell = thermal_conductivity(water)
        d_cell = moisture_diffusivity(water, temp)
        temp_air = linear_interp(t_now, boundary.times, boundary.temperatures)
        water_air = linear_interp(t_now, boundary.times, boundary.moisture)
        temp_surf = surface_value_from_last_cell(
            float(temp[-1]),
            float(k_cell[-1]),
            heat_transfer_coefficient,
            temp_air,
            mesh.spacing,
        )
        water_surf = surface_value_from_last_cell(
            float(water[-1]),
            float(d_cell[-1]),
            mass_transfer_coefficient,
            water_air,
            mesh.spacing,
        )

        heat_rate = float(
            np.sum(
                rho * cp * mesh.volume_factors
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
        iter_log.append(iteration)
        max_temp_err = max(max_temp_err, err_t)
        max_moisture_err = max(max_moisture_err, err_c)
        rho_min = min(rho_min, float(np.min(rho)))
        rho_max = max(rho_max, float(np.max(rho)))
        cp_min = min(cp_min, float(np.min(cp)))
        cp_max = max(cp_max, float(np.max(cp)))
        k_min = min(k_min, float(np.min(k_cell)))
        k_max = max(k_max, float(np.max(k_cell)))
        d_min = min(d_min, float(np.min(d_cell)))
        d_max = max(d_max, float(np.max(d_cell)))

        if step % save_every == 0:
            time_log.append(t_now)
            temp_log.append(
                reconstruct_output_profile(
                    temp,
                    mesh,
                    OUTPUT_RADII,
                    float(k_cell[-1]),
                    heat_transfer_coefficient,
                    temp_air,
                )
            )
            water_log.append(
                reconstruct_output_profile(
                    water,
                    mesh,
                    OUTPUT_RADII,
                    float(d_cell[-1]),
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
        "max_final_temperature_error": float(max_temp_err),
        "max_final_moisture_error": float(max_moisture_err),
        "max_heat_balance_residual": float(max_heat_err),
        "max_moisture_balance_residual": float(max_water_err),
        "minimum_density": rho_min,
        "maximum_density": rho_max,
        "minimum_heat_capacity": cp_min,
        "maximum_heat_capacity": cp_max,
        "minimum_conductivity": k_min,
        "maximum_conductivity": k_max,
        "minimum_diffusivity": d_min,
        "maximum_diffusivity": d_max,
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


def solve_problem_two(
    end_time: float = END_TIME,
    time_step: float = TIME_STEP,
    radial_intervals: int = INTERNAL_CELL_COUNT,
    heat_transfer_coefficient: float = CONVECTIVE_HEAT_COEFF,
    mass_transfer_coefficient: float = CONVECTIVE_MASS_COEFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    result = simulate_problem_two(
        end_time=end_time,
        time_step=time_step,
        cell_count=radial_intervals,
        output_interval=OUTPUT_TIME_INTERVAL,
        heat_transfer_coefficient=heat_transfer_coefficient,
        mass_transfer_coefficient=mass_transfer_coefficient,
    )
    return (
        result.times,
        result.output_radii,
        result.temperature_field,
        result.moisture_field,
        result.diagnostics,
    )


def write_result_workbook(
    times: np.ndarray,
    output_radii: np.ndarray,
    temperature_field: np.ndarray,
    moisture_field: np.ndarray,
) -> None:
    book = load_workbook(OUTPUT_FILE)
    if len(book.worksheets) < 2:
        raise ValueError("result2.xlsx 必须包含温度和水分浓度两个工作表。")

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
    space_coarse = simulate_problem_two(cell_count=80)
    time_fine = simulate_problem_two(cell_count=160, time_step=0.5)

    print("问题二数值检验：")
    report_largest_difference(
        "N=80 与 N=160 的温度场",
        reference.times,
        reference.output_radii,
        space_coarse.temperature_field,
        reference.temperature_field,
    )
    report_largest_difference(
        "N=80 与 N=160 的含水率场",
        reference.times,
        reference.output_radii,
        space_coarse.moisture_field,
        reference.moisture_field,
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
    parser = argparse.ArgumentParser(description="求解问题二的变物性径向传热传质模型")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="额外运行全 3 h 空间和时间步长检验，只在终端输出诊断",
    )
    args = parser.parse_args()

    result = simulate_problem_two()
    write_result_workbook(
        result.times,
        result.output_radii,
        result.temperature_field,
        result.moisture_field,
    )
    print(f"问题二计算完成，结果已写入：{OUTPUT_FILE}")
    if args.validate:
        run_numerical_validation(result)


if __name__ == "__main__":
    main()
