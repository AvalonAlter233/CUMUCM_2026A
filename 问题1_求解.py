"""问题一：固定圆柱药材的径向传热—传质计算。

只读取：附件/附件1.xlsx、附件/附件3/result1.xlsx
只写入：附件/附件3/result1.xlsx

模型：一维轴对称 Fourier 导热 + Fick 非线性水分扩散。
空间离散：有限体积法；时间离散：后向 Euler；非线性：Picard 迭代。
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from openpyxl import load_workbook


# 路径以脚本所在目录为基准
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "附件" / "附件1.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "附件" / "附件3" / "result1.xlsx"

# 药材物性
PELLET_RADIUS = 0.02            # 药材半径，单位 m
PELLET_DENSITY = 820.0          # 药材密度，单位 kg/m^3
PELLET_HEAT_CAPACITY = 2600.0   # 比热容，单位 J/(kg·K)
THERMAL_CONDUCTIVITY = 0.36     # 导热系数，单位 W/(m·K)
CONVECTIVE_HEAT_COEFF = 25.0    # 对流换热系数，单位 W/(m^2·K)
CONVECTIVE_MASS_COEFF = 8.0e-7  # 对流传质系数，单位 m/s

# 初始条件
INITIAL_TEMPERATURE = 28.0      # 初始温度，单位 ℃
INITIAL_MOISTURE = 2.55         # 初始含水率（干基），单位 kg/kg

# 时间推进与非线性迭代设置
END_TIME = 1800                 # 计算结束时间，单位 s
TIME_STEP = 1.0                 # 时间步长，单位 s
RADIAL_INTERVALS = 20           # 径向区间数，20 个区间对应 0.1 cm 空间间隔
CONVERGENCE_TOL = 1.0e-10       # Picard 迭代的相对误差容差
MAX_ITERATIONS = 30             # Picard 迭代次数上限


class DryingRoomBoundary(NamedTuple):
    """附件一给出的烘房侧边界时间序列。"""

    times: np.ndarray            # 时间，单位 s
    temperatures: np.ndarray     # 烘房温度，单位 ℃
    moisture: np.ndarray         # 烘房水分浓度


class RadialGrid(NamedTuple):
    """径向有限体积网格的几何量。

    nodes：       节点半径，共 RADIAL_INTERVALS + 1 个；
    interfaces：  相邻节点中点的半径，共 RADIAL_INTERVALS 个；
    area_factors：单位高度控制体的侧面积 0.5*(r外^2 - r内^2)。
    """

    nodes: np.ndarray
    interfaces: np.ndarray
    area_factors: np.ndarray


def read_drying_boundary(path: Path) -> DryingRoomBoundary:
    """读取附件一中的时间、烘房温度和烘房水分浓度。"""
    workbook = load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook.active
    rows = [
        row for row in worksheet.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    data = np.asarray(rows, dtype=float)

    if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] < 2:
        raise ValueError("附件一必须包含时间、温度、水分浓度三列有效数据。")

    times = data[:, 0]
    if not np.all(np.diff(times) > 0):
        raise ValueError("附件一中的时间必须严格递增。")
    return DryingRoomBoundary(
        times=times,
        temperatures=data[:, 1],
        moisture=data[:, 2],
    )


def linear_interp(t: float, times: np.ndarray, values: np.ndarray) -> float:
    """对附件一的边界数据在时刻 t 处做分段线性插值。"""
    return float(np.interp(t, times, values))


def moisture_diffusivity(moisture: np.ndarray) -> np.ndarray:
    """水分扩散系数 D(C)，单位 m^2/s。"""
    safe_moisture = np.maximum(moisture, 1.0e-12)
    return 7.0e-9 * np.exp(-0.89 / safe_moisture)


def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Thomas 算法：求解三对角线性方程组。"""
    diag = diag.astype(float, copy=True)
    rhs = rhs.astype(float, copy=True)
    upper = upper.astype(float, copy=True)

    for i in range(1, len(diag)):
        factor = lower[i - 1] / diag[i - 1]
        diag[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]

    solution = np.empty_like(rhs)
    solution[-1] = rhs[-1] / diag[-1]
    for i in range(len(diag) - 2, -1, -1):
        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diag[i]
    return solution


def build_radial_grid() -> RadialGrid:
    """建立节点、内部界面和中点控制体面积因子。"""
    nodes = np.linspace(0.0, PELLET_RADIUS, RADIAL_INTERVALS + 1)
    interfaces = (nodes[:-1] + nodes[1:]) / 2.0

    area_factors = np.empty(RADIAL_INTERVALS + 1)
    area_factors[0] = 0.5 * interfaces[0] ** 2
    area_factors[1:-1] = 0.5 * (interfaces[1:] ** 2 - interfaces[:-1] ** 2)
    area_factors[-1] = 0.5 * (PELLET_RADIUS**2 - interfaces[-1] ** 2)
    return RadialGrid(
        nodes=nodes,
        interfaces=interfaces,
        area_factors=area_factors,
    )


def assemble_control_volume_system(
    storage_coeff: np.ndarray,
    old_value: np.ndarray,
    grid: RadialGrid,
    interface_conductivity: np.ndarray,
    surface_transfer_coeff: float,
    boundary_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """组装轴对称控制体的后向 Euler 三对角方程组。

    interface_conductivity 为各内部界面上的传输系数（导热系数或扩散系数）。
    末节点按 Robin 条件 R*Γ*φ_r = -R*β*(φ_s - φ_∞) 加入表面传输项。

    返回 (lower, diag, upper, rhs)。
    """
    n_nodes = len(grid.nodes)
    spacing = grid.nodes[1] - grid.nodes[0]

    lower = np.zeros(n_nodes - 1)
    upper = np.zeros(n_nodes - 1)
    diag = storage_coeff.copy()
    rhs = storage_coeff * old_value

    for i in range(n_nodes):
        inner_coeff = (
            grid.interfaces[i - 1] * interface_conductivity[i - 1] / spacing
            if i > 0 else 0.0
        )
        outer_coeff = (
            grid.interfaces[i] * interface_conductivity[i] / spacing
            if i < n_nodes - 1 else 0.0
        )
        diag[i] += inner_coeff + outer_coeff
        if i > 0:
            lower[i - 1] = -inner_coeff
        if i < n_nodes - 1:
            upper[i] = -outer_coeff

    # 温度与含水率的表面对流边界条件同形：
    #   R*k*T_r = -R*h*(T_s - T_inf)
    #   R*D*C_r = -R*h_m*(C_s - C_inf)
    surface_coeff = PELLET_RADIUS * surface_transfer_coeff
    diag[-1] += surface_coeff
    rhs[-1] += surface_coeff * boundary_value
    return lower, diag, upper, rhs


def advance_temperature(
    temperature: np.ndarray,
    new_time: float,
    grid: RadialGrid,
    boundary: DryingRoomBoundary,
) -> np.ndarray:
    """后向 Euler 推进一个温度时间步。"""
    heat_storage_coeff = (
        PELLET_DENSITY * PELLET_HEAT_CAPACITY * grid.area_factors / TIME_STEP
    )
    interface_conductivity = np.full(RADIAL_INTERVALS, THERMAL_CONDUCTIVITY)

    lower, diag, upper, rhs = assemble_control_volume_system(
        storage_coeff=heat_storage_coeff,
        old_value=temperature,
        grid=grid,
        interface_conductivity=interface_conductivity,
        surface_transfer_coeff=CONVECTIVE_HEAT_COEFF,
        boundary_value=linear_interp(
            new_time, boundary.times, boundary.temperatures
        ),
    )
    return solve_tridiagonal(lower, diag, upper, rhs)


def advance_moisture(
    moisture: np.ndarray,
    new_time: float,
    grid: RadialGrid,
    boundary: DryingRoomBoundary,
) -> tuple[np.ndarray, int]:
    """Picard 迭代推进一个含水率时间步，返回新含水率与迭代次数。"""
    storage_coeff = grid.area_factors / TIME_STEP
    old_moisture = moisture.copy()
    iterate_moisture = moisture.copy()
    boundary_moisture = linear_interp(new_time, boundary.times, boundary.moisture)

    for iteration in range(1, MAX_ITERATIONS + 1):
        diffusivity = moisture_diffusivity(iterate_moisture)
        interface_diffusivity = (
            2.0 * diffusivity[:-1] * diffusivity[1:]
            / np.maximum(diffusivity[:-1] + diffusivity[1:], 1.0e-30)
        )

        lower, diag, upper, rhs = assemble_control_volume_system(
            storage_coeff=storage_coeff,
            old_value=old_moisture,
            grid=grid,
            interface_conductivity=interface_diffusivity,
            surface_transfer_coeff=CONVECTIVE_MASS_COEFF,
            boundary_value=boundary_moisture,
        )
        new_moisture = solve_tridiagonal(lower, diag, upper, rhs)

        rel_error = np.max(
            np.abs(new_moisture - iterate_moisture) / (1.0 + np.abs(new_moisture))
        )
        iterate_moisture = new_moisture
        if rel_error < CONVERGENCE_TOL:
            return new_moisture, iteration

    raise RuntimeError(
        f"含水率Picard迭代未收敛，最后相对误差为{rel_error:.3e}。"
    )


def solve_problem_one() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 1–1800 s 的温度场和含水率场。

    返回 (times, nodes, temperature_field, moisture_field)，
    两个场数据的形状均为 (时刻数, 节点数)。
    """
    boundary = read_drying_boundary(INPUT_FILE)
    grid = build_radial_grid()

    temperature = np.full(len(grid.nodes), INITIAL_TEMPERATURE, dtype=float)
    moisture = np.full(len(grid.nodes), INITIAL_MOISTURE, dtype=float)
    times, temperatures, moistures = [], [], []

    n_steps = int(round(END_TIME / TIME_STEP))
    for step in range(1, n_steps + 1):
        current_time = step * TIME_STEP
        temperature = advance_temperature(temperature, current_time, grid, boundary)
        moisture, _ = advance_moisture(moisture, current_time, grid, boundary)

        times.append(current_time)
        temperatures.append(temperature.copy())
        moistures.append(moisture.copy())

    return (
        np.asarray(times),
        grid.nodes,
        np.asarray(temperatures),
        np.asarray(moistures),
    )


def write_result_workbook(
    times: np.ndarray,
    nodes: np.ndarray,
    temperature_field: np.ndarray,
    moisture_field: np.ndarray,
) -> None:
    """将结果写入附件三的 result1.xlsx，不生成其他结果文件。"""
    workbook = load_workbook(OUTPUT_FILE)
    if len(workbook.worksheets) < 2:
        raise ValueError("result1.xlsx必须包含温度和水分浓度两个工作表。")

    for worksheet, field in zip(
        workbook.worksheets[:2], (temperature_field, moisture_field)
    ):
        # 清除模板示例数据，保留工作表和基本格式。
        if worksheet.max_row > 1:
            worksheet.delete_rows(2, worksheet.max_row - 1)

        worksheet.cell(1, 1).value = "时间\\到药材中心的距离"
        worksheet.cell(1, 1).number_format = "@"
        for col, radius in enumerate(nodes, start=2):
            worksheet.cell(1, col).value = round(float(radius * 100.0), 1)
            worksheet.cell(1, col).number_format = "0.0"

        for row, t in enumerate(times, start=2):
            worksheet.cell(row, 1).value = int(round(float(t)))
            worksheet.cell(row, 1).number_format = "0"
            for col, value in enumerate(field[row - 2], start=2):
                worksheet.cell(row, col).value = round(float(value), 4)
                worksheet.cell(row, col).number_format = "0.0000"

    workbook.save(OUTPUT_FILE)


def main() -> None:
    times, nodes, temperature_field, moisture_field = solve_problem_one()
    write_result_workbook(times, nodes, temperature_field, moisture_field)
    print(f"问题一计算完成，结果已写入：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
