"""问题二：固定圆柱药材的变物性一维径向传热传质计算。

输入：附件/附件1.xlsx、附件/附件3/result2.xlsx
输出：附件/附件3/result2.xlsx

模型约定：
1. C 为药材干基含水率，题给 D(C,T) 视为与该变量配套的有效扩散系数。
2. ρ(C) 作为温度方程中的等效体积密度，不机械乘入水分扩散方程。
3. 附件一水分浓度作为题设等效外界水分变量直接用于 Robin 边界。
4. 附录 3 未重给 h 和 h_m，基线沿用问题一数值；二者可由函数参数改变以做敏感性分析。
5. 耦合仅由状态相关物性构成；因缺少潜热等参数，不加入相变热源和 Soret 项。

数值方法：单元中心有限体积法、后向 Euler、欠松弛 Picard 迭代。
内部计算网格独立于题目要求的 0.1 cm 输出网格；外表面同时计入
最后半个控制体的内部传递阻力和 Robin 边界阻力。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
from openpyxl import load_workbook


# 路径以脚本所在目录为基准，换电脑或换工作目录时无需修改
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result2.xlsx"

# 药材几何、边界传递参数与初始状态
PELLET_RADIUS = 0.02             # 药材半径，单位 m
CONVECTIVE_HEAT_COEFF = 25.0     # 沿用问题一，单位 W/(m^2·K)
CONVECTIVE_MASS_COEFF = 8.0e-7   # 沿用问题一，单位 m/s
INITIAL_TEMPERATURE = 28.0       # 初始温度，单位 ℃
INITIAL_MOISTURE = 2.55          # 初始干基含水率，单位 kg/kg

# 时间、内部空间网格和输出设置
END_TIME = 10800.0               # 计算终点，单位 s（3 h）
TIME_STEP = 1.0                  # 内部时间步长，单位 s
INTERNAL_CELL_COUNT = 160        # 内部有限体积单元数，不等于输出点数
OUTPUT_TIME_INTERVAL = 1.0       # 结果输出间隔，单位 s
OUTPUT_RADII = np.linspace(0.0, PELLET_RADIUS, 21)  # 每隔 0.1 cm 输出

# 非线性迭代设置
CONVERGENCE_TOL = 1.0e-9
MAX_ITERATIONS = 60
RELAXATION_FACTOR = 0.8


class DryingRoomBoundary(NamedTuple):
    """附件一给出的烘房边界时间序列。"""

    times: np.ndarray
    temperatures: np.ndarray
    moisture: np.ndarray


class RadialGrid(NamedTuple):
    """单元中心径向网格及柱壳积分权重。"""

    centers: np.ndarray
    faces: np.ndarray
    volume_factors: np.ndarray
    spacing: float


class SimulationDetails(NamedTuple):
    """内部计算结果及诊断量。"""

    times: np.ndarray
    output_radii: np.ndarray
    temperature_field: np.ndarray
    moisture_field: np.ndarray
    average_temperature: np.ndarray
    average_moisture: np.ndarray
    diagnostics: dict[str, float]


def read_drying_boundary(path: Path) -> DryingRoomBoundary:
    """读取附件一，并检查前三列是否为有效递增时间序列。"""
    workbook = load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook.active
    rows = [
        row for row in worksheet.iter_rows(min_row=2, values_only=True)
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
    """对附件一边界数据进行分段线性插值，不允许外推。"""
    if t < times[0] or t > times[-1]:
        raise ValueError(
            f"计算时刻 {t:g} s 超出附件一范围 "
            f"[{times[0]:g}, {times[-1]:g}] s。"
        )
    return float(np.interp(t, times, values))


def density(moisture: np.ndarray) -> np.ndarray:
    """附录 3 等效密度关系，单位 kg/m^3。"""
    return 650.0 + 128.0 * moisture


def heat_capacity(moisture: np.ndarray) -> np.ndarray:
    """附录 3 比热容关系，单位 J/(kg·K)。"""
    return 1450.0 + 2736.0 * moisture / (moisture + 1.0)


def thermal_conductivity(moisture: np.ndarray) -> np.ndarray:
    """附录 3 导热系数关系，单位 W/(m·K)。"""
    return 0.21 + 0.38 * moisture / (moisture + 1.0)


def moisture_diffusivity(
    moisture: np.ndarray,
    temperature_celsius: np.ndarray,
) -> np.ndarray:
    """附录 3 有效水分扩散系数，温度按 K 代入。"""
    safe_moisture = np.maximum(moisture, 1.0e-12)
    temperature_kelvin = temperature_celsius + 273.15
    return (
        2.4e-3
        * np.exp(-0.45 / safe_moisture)
        * np.exp(-3850.0 / temperature_kelvin)
    )


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """用调和平均计算相邻控制体界面处的传输系数。"""
    denominator = np.maximum(left + right, 1.0e-30)
    return 2.0 * left * right / denominator


def build_radial_grid(cell_count: int) -> RadialGrid:
    """建立等距单元中心有限体积网格。"""
    if cell_count < 2:
        raise ValueError("径向有限体积单元数至少为 2。")
    faces = np.linspace(0.0, PELLET_RADIUS, cell_count + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    volume_factors = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    return RadialGrid(
        centers=centers,
        faces=faces,
        volume_factors=volume_factors,
        spacing=float(faces[1] - faces[0]),
    )


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Thomas 算法求解三对角线性方程组。"""
    diagonal = diagonal.astype(float, copy=True)
    right_hand_side = right_hand_side.astype(float, copy=True)
    upper = upper.astype(float, copy=True)

    for i in range(1, len(diagonal)):
        if abs(diagonal[i - 1]) < 1.0e-30:
            raise FloatingPointError("三对角方程组出现近零主对角元。")
        factor = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        right_hand_side[i] -= factor * right_hand_side[i - 1]

    solution = np.empty_like(right_hand_side)
    solution[-1] = right_hand_side[-1] / diagonal[-1]
    for i in range(len(diagonal) - 2, -1, -1):
        solution[i] = (
            right_hand_side[i] - upper[i] * solution[i + 1]
        ) / diagonal[i]
    return solution


def effective_surface_transfer(
    internal_coefficient: float,
    boundary_coefficient: float,
    half_cell_width: float,
) -> float:
    """合并表面半单元内部阻力与 Robin 边界阻力。"""
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
    """组装变物性单元中心有限体积的后向 Euler 方程组。"""
    cell_count = len(grid.centers)
    lower = np.zeros(cell_count - 1)
    upper = np.zeros(cell_count - 1)
    diagonal = storage_coefficients.copy()
    right_hand_side = storage_coefficients * old_values

    conductance = grid.faces[1:-1] * interface_coefficients / grid.spacing
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    upper[:] = -conductance
    lower[:] = -conductance

    effective_coefficient = effective_surface_transfer(
        internal_coefficient=surface_internal_coefficient,
        boundary_coefficient=surface_transfer_coefficient,
        half_cell_width=0.5 * grid.spacing,
    )
    surface_conductance = PELLET_RADIUS * effective_coefficient
    diagonal[-1] += surface_conductance
    right_hand_side[-1] += surface_conductance * boundary_value
    return lower, diagonal, upper, right_hand_side


def surface_value_from_last_cell(
    last_cell_value: float,
    internal_coefficient: float,
    boundary_coefficient: float,
    boundary_value: float,
    grid_spacing: float,
) -> float:
    """根据半单元内部阻力和对流阻力重构真实表面值。"""
    internal_conductance = 2.0 * internal_coefficient / grid_spacing
    return float(
        (internal_conductance * last_cell_value + boundary_coefficient * boundary_value)
        / (internal_conductance + boundary_coefficient)
    )


def reconstruct_output_profile(
    cell_values: np.ndarray,
    grid: RadialGrid,
    output_radii: np.ndarray,
    surface_internal_coefficient: float,
    surface_transfer_coefficient: float,
    boundary_value: float,
) -> np.ndarray:
    """将单元中心解重构并插值到题目指定的径向位置。"""
    first_radius_squared = grid.centers[0] ** 2
    second_radius_squared = grid.centers[1] ** 2
    center_value = (
        second_radius_squared * cell_values[0]
        - first_radius_squared * cell_values[1]
    ) / (second_radius_squared - first_radius_squared)
    surface_value = surface_value_from_last_cell(
        float(cell_values[-1]),
        surface_internal_coefficient,
        surface_transfer_coefficient,
        boundary_value,
        grid.spacing,
    )
    reconstruction_radii = np.concatenate(([0.0], grid.centers, [PELLET_RADIUS]))
    reconstruction_values = np.concatenate(([center_value], cell_values, [surface_value]))
    return np.interp(output_radii, reconstruction_radii, reconstruction_values)


def radial_volume_average(cell_values: np.ndarray, grid: RadialGrid) -> float:
    """计算圆柱截面的体积加权平均值。"""
    return float(2.0 * np.sum(grid.volume_factors * cell_values) / PELLET_RADIUS**2)


def relative_balance_residual(storage_rate: float, outward_flux: float) -> float:
    """计算当前离散控制方程的积分无量纲残差。"""
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
    """在当前 Picard 含水率下求温度候选解。"""
    rho = density(reference_moisture)
    cp = heat_capacity(reference_moisture)
    conductivity = thermal_conductivity(reference_moisture)
    storage_coefficients = rho * cp * grid.volume_factors / time_step
    interface_conductivity = harmonic_mean(conductivity[:-1], conductivity[1:])
    room_temperature = linear_interp(current_time, boundary.times, boundary.temperatures)

    system = assemble_control_volume_system(
        storage_coefficients,
        old_temperature,
        grid,
        interface_conductivity,
        float(conductivity[-1]),
        heat_transfer_coefficient,
        room_temperature,
    )
    return solve_tridiagonal(*system)


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
    """在当前 Picard 温度、含水率下求水分候选解。"""
    diffusivity = moisture_diffusivity(reference_moisture, reference_temperature)
    interface_diffusivity = harmonic_mean(diffusivity[:-1], diffusivity[1:])
    storage_coefficients = grid.volume_factors / time_step
    room_moisture = linear_interp(current_time, boundary.times, boundary.moisture)

    system = assemble_control_volume_system(
        storage_coefficients,
        old_moisture,
        grid,
        interface_diffusivity,
        float(diffusivity[-1]),
        mass_transfer_coefficient,
        room_moisture,
    )
    return solve_tridiagonal(*system)


def simulate_problem_two(
    end_time: float = END_TIME,
    time_step: float = TIME_STEP,
    cell_count: int = INTERNAL_CELL_COUNT,
    output_interval: float = OUTPUT_TIME_INTERVAL,
    heat_transfer_coefficient: float = CONVECTIVE_HEAT_COEFF,
    mass_transfer_coefficient: float = CONVECTIVE_MASS_COEFF,
) -> SimulationDetails:
    """完成问题二计算，并返回题目网格上的场与数值诊断。"""
    if min(end_time, time_step, output_interval) <= 0.0:
        raise ValueError("终止时间、时间步长和输出间隔必须为正。")
    step_count = int(round(end_time / time_step))
    output_stride = int(round(output_interval / time_step))
    if not np.isclose(step_count * time_step, end_time):
        raise ValueError("终止时间必须是时间步长的整数倍。")
    if output_stride < 1 or not np.isclose(output_stride * time_step, output_interval):
        raise ValueError("输出间隔必须是时间步长的整数倍。")

    boundary = read_drying_boundary(INPUT_FILE)
    if end_time > boundary.times[-1]:
        raise ValueError("附件一边界数据没有覆盖要求的计算时段。")
    grid = build_radial_grid(cell_count)

    temperature = np.full(cell_count, INITIAL_TEMPERATURE, dtype=float)
    moisture = np.full(cell_count, INITIAL_MOISTURE, dtype=float)

    output_times: list[float] = []
    output_temperatures: list[np.ndarray] = []
    output_moistures: list[np.ndarray] = []
    average_temperatures: list[float] = []
    average_moistures: list[float] = []
    iteration_counts: list[int] = []
    max_final_temperature_error = 0.0
    max_final_moisture_error = 0.0
    max_heat_balance_residual = 0.0
    max_moisture_balance_residual = 0.0
    minimum_density = float("inf")
    maximum_density = -float("inf")
    minimum_heat_capacity = float("inf")
    maximum_heat_capacity = -float("inf")
    minimum_conductivity = float("inf")
    maximum_conductivity = -float("inf")
    minimum_diffusivity = float("inf")
    maximum_diffusivity = -float("inf")

    for step in range(1, step_count + 1):
        current_time = step * time_step
        old_temperature = temperature.copy()
        old_moisture = moisture.copy()
        temperature_iterate = temperature.copy()
        moisture_iterate = moisture.copy()

        for iteration in range(1, MAX_ITERATIONS + 1):
            temperature_candidate = solve_temperature_candidate(
                old_temperature,
                moisture_iterate,
                current_time,
                time_step,
                grid,
                boundary,
                heat_transfer_coefficient,
            )
            moisture_candidate = solve_moisture_candidate(
                old_moisture,
                temperature_candidate,
                moisture_iterate,
                current_time,
                time_step,
                grid,
                boundary,
                mass_transfer_coefficient,
            )
            new_temperature = (
                RELAXATION_FACTOR * temperature_candidate
                + (1.0 - RELAXATION_FACTOR) * temperature_iterate
            )
            new_moisture = (
                RELAXATION_FACTOR * moisture_candidate
                + (1.0 - RELAXATION_FACTOR) * moisture_iterate
            )
            temperature_error = float(
                np.max(
                    np.abs(new_temperature - temperature_iterate)
                    / (1.0 + np.abs(new_temperature))
                )
            )
            moisture_error = float(
                np.max(
                    np.abs(new_moisture - moisture_iterate)
                    / (1.0 + np.abs(new_moisture))
                )
            )
            temperature_iterate = new_temperature
            moisture_iterate = new_moisture
            if max(temperature_error, moisture_error) < CONVERGENCE_TOL:
                break
        else:
            raise RuntimeError(
                f"{current_time:g} s 的热湿 Picard 迭代未收敛，"
                f"温度变化量={temperature_error:.3e}，"
                f"含水率变化量={moisture_error:.3e}。"
            )

        temperature = temperature_iterate
        moisture = moisture_iterate
        if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(moisture)):
            raise FloatingPointError(f"{current_time:g} s 出现非有限场变量。")
        if np.min(moisture) <= 0.0:
            raise FloatingPointError(f"{current_time:g} s 出现非正含水率。")

        rho = density(moisture)
        cp = heat_capacity(moisture)
        conductivity = thermal_conductivity(moisture)
        diffusivity = moisture_diffusivity(moisture, temperature)
        room_temperature = linear_interp(current_time, boundary.times, boundary.temperatures)
        room_moisture = linear_interp(current_time, boundary.times, boundary.moisture)
        surface_temperature = surface_value_from_last_cell(
            float(temperature[-1]),
            float(conductivity[-1]),
            heat_transfer_coefficient,
            room_temperature,
            grid.spacing,
        )
        surface_moisture = surface_value_from_last_cell(
            float(moisture[-1]),
            float(diffusivity[-1]),
            mass_transfer_coefficient,
            room_moisture,
            grid.spacing,
        )

        # 检查当前离散 PDE 的积分残差，而不是把变热容材料的总显热差分
        # 误当成方程左端。
        heat_storage_rate = float(
            np.sum(
                rho * cp * grid.volume_factors
                * (temperature - old_temperature) / time_step
            )
        )
        moisture_storage_rate = float(
            np.sum(grid.volume_factors * (moisture - old_moisture) / time_step)
        )
        heat_flux_outward = (
            PELLET_RADIUS * heat_transfer_coefficient
            * (surface_temperature - room_temperature)
        )
        moisture_flux_outward = (
            PELLET_RADIUS * mass_transfer_coefficient
            * (surface_moisture - room_moisture)
        )
        max_heat_balance_residual = max(
            max_heat_balance_residual,
            relative_balance_residual(heat_storage_rate, heat_flux_outward),
        )
        max_moisture_balance_residual = max(
            max_moisture_balance_residual,
            relative_balance_residual(moisture_storage_rate, moisture_flux_outward),
        )
        iteration_counts.append(iteration)
        max_final_temperature_error = max(max_final_temperature_error, temperature_error)
        max_final_moisture_error = max(max_final_moisture_error, moisture_error)
        minimum_density = min(minimum_density, float(np.min(rho)))
        maximum_density = max(maximum_density, float(np.max(rho)))
        minimum_heat_capacity = min(minimum_heat_capacity, float(np.min(cp)))
        maximum_heat_capacity = max(maximum_heat_capacity, float(np.max(cp)))
        minimum_conductivity = min(minimum_conductivity, float(np.min(conductivity)))
        maximum_conductivity = max(maximum_conductivity, float(np.max(conductivity)))
        minimum_diffusivity = min(minimum_diffusivity, float(np.min(diffusivity)))
        maximum_diffusivity = max(maximum_diffusivity, float(np.max(diffusivity)))

        if step % output_stride == 0:
            output_times.append(current_time)
            output_temperatures.append(
                reconstruct_output_profile(
                    temperature,
                    grid,
                    OUTPUT_RADII,
                    float(conductivity[-1]),
                    heat_transfer_coefficient,
                    room_temperature,
                )
            )
            output_moistures.append(
                reconstruct_output_profile(
                    moisture,
                    grid,
                    OUTPUT_RADII,
                    float(diffusivity[-1]),
                    mass_transfer_coefficient,
                    room_moisture,
                )
            )
            average_temperatures.append(radial_volume_average(temperature, grid))
            average_moistures.append(radial_volume_average(moisture, grid))

    diagnostics = {
        "internal_cell_count": float(cell_count),
        "time_step": float(time_step),
        "max_picard_iterations": float(max(iteration_counts)),
        "mean_picard_iterations": float(np.mean(iteration_counts)),
        "max_final_temperature_error": float(max_final_temperature_error),
        "max_final_moisture_error": float(max_final_moisture_error),
        "max_heat_balance_residual": float(max_heat_balance_residual),
        "max_moisture_balance_residual": float(max_moisture_balance_residual),
        "minimum_density": minimum_density,
        "maximum_density": maximum_density,
        "minimum_heat_capacity": minimum_heat_capacity,
        "maximum_heat_capacity": maximum_heat_capacity,
        "minimum_conductivity": minimum_conductivity,
        "maximum_conductivity": maximum_conductivity,
        "minimum_diffusivity": minimum_diffusivity,
        "maximum_diffusivity": maximum_diffusivity,
    }
    return SimulationDetails(
        times=np.asarray(output_times),
        output_radii=OUTPUT_RADII.copy(),
        temperature_field=np.asarray(output_temperatures),
        moisture_field=np.asarray(output_moistures),
        average_temperature=np.asarray(average_temperatures),
        average_moisture=np.asarray(average_moistures),
        diagnostics=diagnostics,
    )


def solve_problem_two(
    end_time: float = END_TIME,
    time_step: float = TIME_STEP,
    radial_intervals: int = INTERNAL_CELL_COUNT,
    heat_transfer_coefficient: float = CONVECTIVE_HEAT_COEFF,
    mass_transfer_coefficient: float = CONVECTIVE_MASS_COEFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """保留原调用接口；radial_intervals 现表示内部有限体积单元数。"""
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
    """按附件三模板写入 result2.xlsx，不生成其他结果文件。"""
    workbook = load_workbook(OUTPUT_FILE)
    if len(workbook.worksheets) < 2:
        raise ValueError("result2.xlsx 必须包含温度和水分浓度两个工作表。")

    for worksheet, field in zip(
        workbook.worksheets[:2], (temperature_field, moisture_field)
    ):
        if worksheet.max_row > 1:
            worksheet.delete_rows(2, worksheet.max_row - 1)

        worksheet.cell(1, 1).value = "时间\\到药材中心的距离"
        worksheet.cell(1, 1).number_format = "@"
        for column, radius in enumerate(output_radii, start=2):
            worksheet.cell(1, column).value = round(float(radius * 100.0), 1)
            worksheet.cell(1, column).number_format = "0.0"

        for row, current_time in enumerate(times, start=2):
            worksheet.cell(row, 1).value = int(round(float(current_time)))
            worksheet.cell(row, 1).number_format = "0"
            for column, value in enumerate(field[row - 2], start=2):
                worksheet.cell(row, column).value = round(float(value), 4)
                worksheet.cell(row, column).number_format = "0.0000"

    workbook.save(OUTPUT_FILE)


def report_largest_difference(
    label: str,
    times: np.ndarray,
    radii: np.ndarray,
    first_field: np.ndarray,
    second_field: np.ndarray,
) -> None:
    """输出两个同网格结果的最大绝对差及其发生位置。"""
    difference = np.abs(first_field - second_field)
    time_index, radius_index = np.unravel_index(
        int(np.argmax(difference)), difference.shape
    )
    print(
        f"{label}：最大绝对差={difference[time_index, radius_index]:.6e}，"
        f"位置为 t={times[time_index]:g} s、"
        f"r={radii[radius_index] * 100:g} cm；"
        f"终点最大差={np.max(difference[-1]):.6e}。"
    )


def run_numerical_validation(reference: SimulationDetails) -> None:
    """复现论文中的全 3 h 空间、时间离散检验，不生成额外文件。"""
    coarse_space = simulate_problem_two(cell_count=80)
    fine_time = simulate_problem_two(cell_count=160, time_step=0.5)

    print("问题二数值检验：")
    report_largest_difference(
        "N=80 与 N=160 的温度场",
        reference.times,
        reference.output_radii,
        coarse_space.temperature_field,
        reference.temperature_field,
    )
    report_largest_difference(
        "N=80 与 N=160 的含水率场",
        reference.times,
        reference.output_radii,
        coarse_space.moisture_field,
        reference.moisture_field,
    )
    report_largest_difference(
        "时间步 1 s 与 0.5 s 的温度场",
        reference.times,
        reference.output_radii,
        reference.temperature_field,
        fine_time.temperature_field,
    )
    report_largest_difference(
        "时间步 1 s 与 0.5 s 的含水率场",
        reference.times,
        reference.output_radii,
        reference.moisture_field,
        fine_time.moisture_field,
    )
    print(f"主计算诊断：{reference.diagnostics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="求解问题二的变物性径向传热传质模型")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="额外运行全 3 h 空间和时间步长检验，只在终端输出诊断",
    )
    arguments = parser.parse_args()

    result = simulate_problem_two()
    write_result_workbook(
        result.times,
        result.output_radii,
        result.temperature_field,
        result.moisture_field,
    )
    print(f"问题二计算完成，结果已写入：{OUTPUT_FILE}")
    if arguments.validate:
        run_numerical_validation(result)


if __name__ == "__main__":
    main()
