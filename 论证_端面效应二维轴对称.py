# -*- coding: utf-8 -*-
"""
A 级论证之一：端面效应的一维假设误差量化（二维轴对称有限体积）

问题一至问题三采用圆柱中部截面的一维径向模型，隐含假设端面传热传质可忽略。
题面本身未禁止端面作用：端面面积占全部外表面积 7.41%（R/L = 0.08）。
本脚本建立一个二维轴对称 (r, z) 有限体积求解器，在同一网格上比较两种端面情形：

    情形 A  端面绝热绝湿（零通量）  —— 应与一维模型一致，作为实现校验
    情形 B  端面与侧面处于相同环境并采用相同 (h, h_m) —— 端面效应的敏感性上界

A 与 B 的达标时间之差即为端面效应对本文结论的影响尺度。
另外把情形 A 与一维高分辨率结果比较，用于报告网格误差。

物理设置与问题三一致：固定半径 R = 2 cm、附录三变物性、烘房边界 4 h 后取末 1 h 均值。

运行：
    python 论证_端面效应二维轴对称.py                 # 默认网格，完整计算
    python 论证_端面效应二维轴对称.py --steps 200     # 仅跑 200 步，用于基准测试

输出：终端报表 + 附件/附件3/论证_端面效应.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# ---------------------------------------------------------------- 题目常量
R_PELLET = 0.02          # 药材半径 [m]
L_PELLET = 0.25          # 药材长度 [m]
LHALF = L_PELLET / 2.0   # 对称面取在中部，计算半长 [m]
H_CONV = 25.0
HM_CONV = 8.0e-7
T0 = 28.0
C0 = 2.55
THRESHOLD = 0.15
T_LONG = 49.9989344
C_LONG = 0.0499875410
T_SWITCH = 14400.0

RHO = lambda C: 650.0 + 128.0 * C
CP = lambda C: 1450.0 + 2736.0 * C / (C + 1.0)
KTH = lambda C: 0.21 + 0.38 * C / (C + 1.0)
DIFF = lambda C, T: 2.4e-3 * np.exp(-0.45 / np.maximum(C, 1e-12)) * np.exp(
    -3850.0 / (T + 273.15))


# ---------------------------------------------------------------- xlsx 读取
def _col_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_sheet(path: str, index: int = 0):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(NS + "si"):
                shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
        names = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))
        root = ET.fromstring(z.read(names[index]))
        rows = []
        for row in root.iter(NS + "row"):
            cells = {}
            for c in row.iter(NS + "c"):
                v = c.find(NS + "v")
                if v is None or v.text is None:
                    continue
                txt = v.text
                if c.get("t") == "s":
                    txt = shared[int(txt)]
                cells[_col_index(c.get("r", "A1"))] = txt
            if cells:
                rows.append([cells.get(i) for i in range(max(cells) + 1)])
        return rows


def load_chamber(root: str):
    rows = read_sheet(os.path.join(root, "附件", "附件1.xlsx"), 0)
    d = np.array([[float(x) for x in r[:3]] for r in rows[1:]
                  if r and r[0] is not None])
    return d[:, 0], d[:, 1], d[:, 2]


def boundary_at(t, t_arr, T_arr, C_arr):
    if t <= T_SWITCH:
        return float(np.interp(t, t_arr, T_arr)), float(np.interp(t, t_arr, C_arr))
    return T_LONG, C_LONG


# ---------------------------------------------------------------- 网格
class Grid:
    """单元中心二维轴对称网格，索引 p = j * Nr + i（j 为轴向，i 为径向）。"""

    def __init__(self, Nr: int, Nz: int):
        self.Nr, self.Nz = Nr, Nz
        self.dr = R_PELLET / Nr
        self.dz = LHALF / Nz
        self.r = (np.arange(Nr) + 0.5) * self.dr            # (Nr,)
        self.z = (np.arange(Nz) + 0.5) * self.dz            # (Nz,)

        self.rf = np.arange(1, Nr) * self.dr                # 内部径向界面 (Nr-1,)
        # 展平顺序固定为 p = j * Nr + i，故 p 处的半径权重为 r[p % Nr]
        self.rg = np.tile(self.r, Nz)
        self.N = Nr * Nz

        jj, ii = np.meshgrid(np.arange(Nz), np.arange(Nr), indexing="ij")
        self.i = ii.ravel()
        self.j = jj.ravel()

        # 邻接掩码（用于把 scipy 对角线的环绕项清零）
        self.has_e = (self.i < Nr - 1)
        self.has_w = (self.i > 0)
        self.has_n = (self.j < Nz - 1)
        self.has_s = (self.j > 0)

        # 谐和平均辅助：把 (N,) 场映射到径向 / 轴向界面
        self.idx_e = np.where(self.has_e, np.arange(self.N) + 1, 0)
        self.idx_n = np.where(self.has_n, np.arange(self.N) + Nr, 0)

    def volume(self):
        return self.rg * self.dr * self.dz


def harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """界面物性谐和平均，避免分母为零。"""
    return 2.0 * a * b / np.maximum(a + b, 1e-300)


# ---------------------------------------------------------------- 求解器
class Solver2D:
    def __init__(self, grid: Grid, exposed_ends: bool, dt: float):
        self.g = grid
        self.exposed = exposed_ends
        self.dt = dt
        self.V = grid.volume()

        # ---- 预计算矩阵稀疏结构（稀疏模式固定，每步只更新 data）----
        n = grid.N
        idx = np.arange(n)
        pe = idx[grid.has_e]          # 有东邻的单元
        pw = idx[grid.has_w]
        pn = idx[grid.has_n]
        ps = idx[grid.has_s]
        rows = np.concatenate([idx, pe, pw, pn, ps])
        cols = np.concatenate([idx, pe + 1, pw - 1, pn + grid.Nr, ps - grid.Nr])
        self.rows, self.cols = rows, cols
        sizes = [len(idx), len(pe), len(pw), len(pn), len(ps)]
        self.slices = np.cumsum([0] + sizes)
        self.pe, self.pw, self.pn, self.ps = pe, pw, pn, ps
        self.shape = (n, n)

    # ---- 组装一个标量场的三对角（五行）隐式方程组 ----
    def assemble(self, s: np.ndarray, gamma: np.ndarray, beta: float,
                 phi_inf: float, phi_old: np.ndarray, dr_face=None):
        """
        s       : (N,) 累积项系数（温度取 rho*cp，水分取 1）
        gamma   : (N,) 传输系数（k 或 D）
        beta    : 对流系数（h 或 h_m）
        phi_inf : 环境值
        返回 CSR 矩阵与右端向量
        """
        g = self.g
        N, Nr, Nz = g.N, g.Nr, g.Nz
        pe, pw, pn, ps = self.pe, self.pw, self.pn, self.ps

        # --- 径向内部面电导（界面位于 i 与 i+1 之间）---
        ge_int = np.zeros(N)
        ge_int[pe] = g.rf[g.i[pe]] * g.dz * harmonic(gamma[pe], gamma[pe + 1]) / g.dr

        # --- 轴向内部面电导（界面位于 j 与 j+1 之间）---
        gn_int = np.zeros(N)
        gn_int[pn] = g.rg[pn] * g.dr * harmonic(gamma[pn], gamma[pn + Nr]) / g.dz

        # --- 边界电导：径向表面 i = Nr-1 ---
        gb = np.zeros(N)
        pR = np.arange(N)[g.i == Nr - 1]
        h_eff_R = 1.0 / (1.0 / beta + g.dr / (2.0 * gamma[pR]))
        gb[pR] += R_PELLET * g.dz * h_eff_R

        # --- 边界电导：轴向端面 j = Nz-1（仅情形 B）---
        if self.exposed:
            pZ = np.arange(N)[g.j == Nz - 1]
            h_eff_Z = 1.0 / (1.0 / beta + g.dz / (2.0 * gamma[pZ]))
            gb[pZ] += g.rg[pZ] * g.dr * h_eff_Z

        # --- 组装 ---
        aP0 = s * self.V / self.dt
        main = aP0 + gb
        np.add.at(main, pe, ge_int[pe])
        np.add.at(main, pw, ge_int[pw - 1])
        np.add.at(main, pn, gn_int[pn])
        np.add.at(main, ps, gn_int[ps - Nr])

        data = np.empty(self.rows.size)
        sl = self.slices
        data[sl[0]:sl[1]] = main
        data[sl[1]:sl[2]] = -ge_int[pe]
        data[sl[2]:sl[3]] = -ge_int[pw - 1]
        data[sl[3]:sl[4]] = -gn_int[pn]
        data[sl[4]:sl[5]] = -gn_int[ps - Nr]

        A = sp.csr_matrix((data, (self.rows, self.cols)), shape=self.shape)
        rhs = aP0 * phi_old + gb * phi_inf
        return A, rhs

    def solve_field(self, s, gamma, beta, phi_inf, phi_old):
        A, rhs = self.assemble(s, gamma, beta, phi_inf, phi_old)
        return spla.spsolve(A, rhs)


def run_case(root: str, grid: Grid, exposed: bool, dt: float, dt_out: float,
             max_steps: int | None, omega: float = 1.0,
             picard_tol: float = 1e-9, verbose: bool = True):
    t_arr, T_arr, C_arr = load_chamber(root)
    solver = Solver2D(grid, exposed, dt)
    N = grid.N

    T = np.full(N, T0)
    C = np.full(N, C0)

    t = 0.0
    t_star = None
    t_star_interp = None
    n_steps = 0
    next_out = dt_out
    history = []           # (t, C_max, T_center, C_surface_proxy)
    prev_cmax = float(C.max())
    prev_t = 0.0

    # 轴心与端面轴心位置（用于监控）
    p_center = 0                      # (j=0, i=0) -> r=0, z=0
    p_mid_surface = (grid.Nz // 2) * grid.Nr + (grid.Nr - 1)

    t_start = time.time()
    while True:
        if max_steps is not None and n_steps >= max_steps:
            break
        T_inf, C_inf = boundary_at(t + dt, t_arr, T_arr, C_arr)

        T_new, C_new = T.copy(), C.copy()
        for _ in range(60):
            rho_cp = RHO(C_new) * CP(C_new)
            k_ = KTH(C_new)
            Tsol = solver.solve_field(rho_cp, k_, H_CONV, T_inf, T)

            D_ = DIFF(C_new, Tsol)
            Csol = solver.solve_field(np.ones(N), D_, HM_CONV, C_inf, C)

            T_next = omega * Tsol + (1.0 - omega) * T_new
            C_next = omega * Csol + (1.0 - omega) * C_new

            dT = np.max(np.abs(T_next - T_new) / (1.0 + np.abs(T_next)))
            dC = np.max(np.abs(C_next - C_new) / (1.0 + np.abs(C_next)))
            T_new, C_new = T_next, C_next
            if dT < picard_tol and dC < picard_tol:
                break

        T, C = T_new, C_new
        t += dt
        n_steps += 1

        if t + 1e-9 >= next_out:
            history.append((t, float(C.max()), float(T[p_center]),
                            float(C[p_mid_surface])))
            next_out += dt_out

        # 达标判定：全域最大含水率穿越 0.15（对相邻两层做线性插值）
        c_max = float(C.max())
        if t_star is None and c_max < THRESHOLD:
            t_star = t
            t_star_interp = prev_t + (prev_cmax - THRESHOLD) / (prev_cmax - c_max) * dt
            if verbose:
                print(f"    [达标] 离散步 t = {t / 3600:.4f} h，"
                      f"插值穿越 t* = {t_star_interp / 3600:.4f} h "
                      f"({n_steps} 步, 用时 {time.time() - t_start:.0f} s)")
            break
        prev_cmax, prev_t = c_max, t

        if verbose and n_steps % 500 == 0:
            el = time.time() - t_start
            print(f"    t = {t / 3600:7.3f} h  C_max = {c_max:.6f}  "
                  f"T_center = {T[p_center]:7.3f}  "
                  f"[{n_steps} 步, {el:.0f} s, {el / n_steps * 1000:.1f} ms/步]")

    elapsed = time.time() - t_start
    return {
        "exposed_ends": exposed,
        "Nr": grid.Nr, "Nz": grid.Nz, "dt_s": dt,
        "n_steps": n_steps,
        "t_star_s": t_star,
        "t_star_h": (t_star / 3600.0) if t_star else None,
        "t_star_interp_s": t_star_interp,
        "t_star_interp_h": (t_star_interp / 3600.0) if t_star_interp else None,
        "final_C_max": float(C.max()),
        "final_T_center": float(T[p_center]),
        "wall_time_s": elapsed,
        "ms_per_step": elapsed / max(n_steps, 1) * 1000.0,
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nr", type=int, default=40)
    ap.add_argument("--Nz", type=int, default=250)
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=None,
                    help="仅推进给定步数（基准测试用）")
    ap.add_argument("--case", choices=["A", "B", "both"], default="both")
    ap.add_argument("--out", type=str, default=None,
                    help="输出 JSON 文件名（并行运行两个情形时分别指定）")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    print("工作目录:", root)
    grid = Grid(args.Nr, args.Nz)
    print(f"网格: Nr={grid.Nr}, Nz={grid.Nz}, N={grid.N} 单元, "
          f"dr={grid.dr * 100:.4f} cm, dz={grid.dz * 100:.4f} cm")
    print(f"物理域: r in [0, {R_PELLET * 100:.1f}] cm, "
          f"z in [0, {LHALF * 100:.3f}] cm (半长), dt={args.dt} s")

    out = {"scope": "端面效应二维轴对称量化（固定半径，附录三物性，问题三边界）",
           "grid": {"Nr": grid.Nr, "Nz": grid.Nz, "N": grid.N,
                    "dr_m": grid.dr, "dz_m": grid.dz},
           "dt_s": args.dt,
           "one_dimensional_reference_h": 57.5224,
           "cases": {}}

    todo = []
    if args.case in ("A", "both"):
        todo.append(("A_insulated_ends", False))
    if args.case in ("B", "both"):
        todo.append(("B_exposed_ends", True))

    for label, exposed in todo:
        print(f"\n{'=' * 78}\n情形 {label}  (端面{'暴露' if exposed else '绝热绝湿'})"
              f"\n{'=' * 78}")
        res = run_case(root, grid, exposed, args.dt, dt_out=60.0,
                       max_steps=args.steps)
        if res["t_star_interp_h"]:
            print(f"  -> 插值穿越 t* = {res['t_star_interp_h']:.4f} h")
        else:
            print(f"  -> 未达标（仅跑了 {res['n_steps']} 步）")
        print(f"     用时 {res['wall_time_s']:.0f} s, {res['ms_per_step']:.1f} ms/步")
        # 轨迹每 600 s 保留一点，控制 JSON 体积
        res["history"] = res["history"][::10]
        out["cases"][label] = res

    # ---------------- 端面效应量 ----
    if "A_insulated_ends" in out["cases"] and "B_exposed_ends" in out["cases"]:
        a = out["cases"]["A_insulated_ends"]["t_star_interp_h"]
        b = out["cases"]["B_exposed_ends"]["t_star_interp_h"]
        if a and b:
            out["end_face_effect"] = {
                "t_star_A_h": a, "t_star_B_h": b,
                "delta_s": (b - a) * 3600.0,
                "delta_h": b - a,
                "relative_percent": (b - a) / a * 100.0,
                "vs_1d_highres_s": (a - 57.5224) * 3600.0,
                "vs_1d_highres_percent": (a - 57.5224) / 57.5224 * 100.0,
            }
            print(f"\n{'=' * 78}\n端面效应\n{'=' * 78}")
            print(f"  情形 A（端面绝热绝湿）t* = {a:.4f} h")
            print(f"  情形 B（端面暴露）    t* = {b:.4f} h")
            print(f"  端面效应   Δt* = {(b - a) * 3600:+.1f} s ({(b - a) / a * 100:+.3f}%)")
            print(f"  情形 A 相对一维高分辨率(57.5224 h)偏差 = "
                  f"{(a - 57.5224) * 3600:+.1f} s ({(a - 57.5224) / 57.5224 * 100:+.3f}%)"
                  f"  <- 径向网格误差")

    outpath = os.path.join(root, "附件", "附件3",
                           args.out or "论证_端面效应.json")
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n完成。结果已写入:", outpath)


if __name__ == "__main__":
    main()
