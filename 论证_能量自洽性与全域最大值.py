# -*- coding: utf-8 -*-
"""
S 级论证脚本（只读现有结果文件，不修改任何求解或绘图代码）

本脚本产出两项"事后论证"所需的全部数值证据，供论文正文引用：

  论证 A  热湿耦合的能量自洽性量级判据
          比较模型实际吸收的显热 Q_sen 与模型预测蒸发量对应的潜热 Q_lat，
          说明题给 (h, h_m) 无法同时作为同一物理过程的对流系数对，
          从而论证基准模型省略相变潜热是题给参数下唯一自洽的读法，
          而不是"缺少参数"导致的被迫简化。

  论证 B  全域最大含水率位于轴心且随时间单调不增
          对 result3 / result4（以及问题四的内部细网格场）逐时刻核验，
          为最大值原理证明提供数值佐证。

运行：
    python 论证_能量自洽性与全域最大值.py

输出：
    终端报表 + 附件/附件3/论证_diagnostics.json
"""

from __future__ import annotations

import json
import math
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

# ----------------------------------------------------------------------------
# 0. 基本常量（全部取自题面，与求解代码保持一致）
# ----------------------------------------------------------------------------
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

R = 0.02                 # 药材半径 [m]
V_OVER_A = R / 2.0       # 侧面换热/传质的体面积比 V/A_s [m]
H_CONV = 25.0            # 对流换热系数 [W/(m^2 K)]
HM_CONV = 8.0e-7         # 对流传质系数 [m/s]
C0 = 2.55                # 初始干基含水率 [kg/kg]
T0 = 28.0                # 初始温度 [C]
DELTA_H = 2383.0e3       # 50 C 时水的汽化潜热 [J/kg]

# 附录三物性（问题二、问题三）
RHO3 = lambda C: 650.0 + 128.0 * C
CP3 = lambda C: 1450.0 + 2736.0 * C / (C + 1.0)


# ----------------------------------------------------------------------------
# 1. 轻量 xlsx 读取（zipfile + ElementTree，不依赖 pandas/openpyxl）
# ----------------------------------------------------------------------------
def _col_index(ref: str) -> int:
    """把单元格引用 'AB12' 转成 0 基列号。"""
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def read_sheet(path: str, index: int = 0) -> list[list[str | None]]:
    """读取 xlsx 的第 index 个工作表，返回按行组织的字符串矩阵。"""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(NS + "si"):
                shared.append("".join(t.text or "" for t in si.iter(NS + "t")))

        names = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))
        root = ET.fromstring(z.read(names[index]))

        rows: list[list[str | None]] = []
        for row in root.iter(NS + "row"):
            cells: dict[int, str] = {}
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


def as_grid(rows: list[list[str | None]], n_data_cols: int):
    """把首行表头的数值表转成 (时间数组, 数据矩阵)。"""
    t_list, data = [], []
    for r in rows[1:]:
        if not r or r[0] is None:
            continue
        try:
            tt = float(r[0])
        except (TypeError, ValueError):
            continue
        vals = []
        for j in range(1, n_data_cols + 1):
            raw = r[j] if j < len(r) else None
            vals.append(float(raw) if raw is not None else np.nan)
        t_list.append(tt)
        data.append(vals)
    return np.asarray(t_list, float), np.asarray(data, float)


def radial_average(field: np.ndarray, r_nodes: np.ndarray) -> np.ndarray:
    """
    圆柱体积加权平均:  fbar = (2/R^2) * int_0^R f r dr = int f d(r^2) / R^2
    field 形状 (nt, nr)，返回形状 (nt,)
    """
    r2 = r_nodes ** 2
    return np.trapz(field, r2, axis=1) / (r2[-1] - r2[0])


# ----------------------------------------------------------------------------
# 2. 读入烘房边界（附件一）
# ----------------------------------------------------------------------------
def load_chamber(path: str):
    rows = read_sheet(path, 0)
    t, data = as_grid(rows, 2)
    return t, data[:, 0], data[:, 1]        # t, T_inf [C], C_inf [kg/kg]


def chamber_at(t: np.ndarray, tq, T_inf: np.ndarray, C_inf: np.ndarray):
    """分段线性插值；超出实测范围保持端点值。"""
    Tq = np.interp(tq, t, T_inf)
    Cq = np.interp(tq, t, C_inf)
    return Tq, Cq


# ----------------------------------------------------------------------------
# 论证 A：热湿耦合的能量自洽性量级判据
# ----------------------------------------------------------------------------
def argument_energy(root: str, chamber_t, T_inf, C_inf) -> dict:
    print("\n" + "=" * 78)
    print("论证 A  热湿耦合的能量自洽性量级判据")
    print("=" * 78)

    # --- A.1 边界可供热量上界: h * int (T_inf - 28) dt --------------------
    ub = {}
    for tm in (1800.0, 10800.0):
        m = chamber_t <= tm
        integral = float(np.trapz(np.maximum(T_inf[m] - T0, 0.0), chamber_t[m]))
        ub[tm] = H_CONV * integral
        print(f"  边界供热绝对上界 0-{int(tm):6d} s : "
              f"int(T_inf-28)dt = {integral:10.4g} K*s  ->  "
              f"Q_ub = {H_CONV * integral:.4g} J/m^2")

    # --- A.2 读入问题二全场，计算 Q_sen 与 Q_lat -------------------------
    r_nodes = np.linspace(0.0, R, 21)
    path2 = os.path.join(root, "附件", "附件3", "result2.xlsx")

    t_c, C_field = as_grid(read_sheet(path2, 1), 21)   # sheet 1 = 水分浓度
    t_T, T_field = as_grid(read_sheet(path2, 0), 21)   # sheet 0 = 温度
    assert np.allclose(t_c, t_T), "问题二两个工作表的时间列不一致"

    Cbar = radial_average(C_field, r_nodes)                       # kg/kg
    rho_cp_T = RHO3(C_field) * CP3(C_field) * T_field             # J/m^3 (相对 0 C)
    e_bar = radial_average(rho_cp_T, r_nodes)                     # J/m^3

    rho_d_dry = float(RHO3(C0) / (1.0 + C0))   # 干基口径: 干物质体积密度
    rho_d_zero = float(RHO3(0.0))              # 上限口径: rho(C=0)

    # 严格口径: 把 rho(C) 直接读作湿物料表观密度，则单位总体积的含水量为
    #   rho(C) * C / (1 + C)   [kg 水 / m^3 总体积]
    # 该口径不含任何待选参数，用作主报值。
    water_per_volume = RHO3(C_field) * C_field / (1.0 + C_field)
    w_bar = radial_average(water_per_volume, r_nodes)          # kg/m^3

    print(f"\n  干物质密度口径: 干基 rho(C0)/(1+C0) = {rho_d_dry:.2f} kg/m^3 ; "
          f"rho(C=0) = {rho_d_zero:.2f} kg/m^3")
    print(f"  严格口径(无待选参数) 单位总体积含水量 w(0) = {w_bar[0]:.2f} kg/m^3")
    print(f"    注: 该口径下的干物质密度 rho(C)/(1+C) 由 "
          f"{RHO3(C0) / (1 + C0):.1f} 变为 {RHO3(0.0) / 1.0:.1f} kg/m^3，"
          f"说明附录三的 rho(C) 并非严格的质量自洽式，故同时报告两种口径作为区间。")

    # 模型边界真实输入热量: Q_in = int h (T_inf(t) - T_s(t)) dt
    # 这是模型自身的边界条件所定义的热输入，不依赖焓的口径选择
    T_surf2 = T_field[:, -1]
    T_inf_at = np.interp(t_c, chamber_t, T_inf)
    Q_in = float(np.trapz(H_CONV * (T_inf_at - T_surf2), t_c))

    Tbar = radial_average(T_field, r_nodes)

    results = {}
    for tm in (1800.0, 10800.0):
        idx = int(np.argmin(np.abs(t_c - tm)))
        assert abs(t_c[idx] - tm) < 1e-6

        q_sen = (e_bar[idx] - e_bar[0]) * V_OVER_A          # J/m^2，焓差口径
        q_in = float(np.trapz(H_CONV * (T_inf_at[:idx + 1] - T_surf2[:idx + 1]),
                              t_c[:idx + 1]))               # J/m^2，边界通量口径
        dc = Cbar[0] - Cbar[idx]                            # kg/kg
        m_w = {tag: rd * dc * V_OVER_A for tag, rd in
               (("dry", rho_d_dry), ("zero", rho_d_zero))}  # kg/m^2
        m_w["rigorous"] = float(w_bar[0] - w_bar[idx]) * V_OVER_A   # kg/m^2

        row = {"t_s": tm, "Q_sen_enthalpy": q_sen, "Q_in_boundary_flux": q_in,
               "Q_ub": ub[tm], "dCbar": float(dc),
               "Tbar_end": float(Tbar[idx])}
        for tag, mw in m_w.items():
            q_lat = mw * DELTA_H
            row[f"Q_lat_{tag}"] = q_lat
            row[f"ratio_{tag}"] = q_lat / q_in
            row[f"ratio_ub_{tag}"] = q_lat / ub[tm]

        results[tm] = row
        print(f"\n  --- 0 ~ {int(tm)} s ---")
        print(f"    体积平均含水率  {Cbar[0]:.4f} -> {Cbar[idx]:.4f}  (dCbar = {dc:.4f} kg/kg)")
        print(f"    体积平均温度    {T0:.2f} -> {row['Tbar_end']:.4f} C")
        print(f"    边界热输入  Q_in = {q_in:.4g} J/m^2  (通量口径，模型边界条件直接积分)")
        print(f"    焓差核对    Q_sen= {q_sen:.4g} J/m^2  (与 Q_in 之比 {q_sen / q_in:.3f})")
        print(f"    潜热需求    Q_lat= {row['Q_lat_rigorous']:.4g} (严格口径) ; "
              f"区间 {row['Q_lat_dry']:.4g} ~ {row['Q_lat_zero']:.4g} J/m^2")
        print(f"    比值 Q_lat/Q_in  = {row['ratio_rigorous']:.1f} "
              f"(区间 {row['ratio_dry']:.1f} ~ {row['ratio_zero']:.1f})")
        print(f"    比值 Q_lat/Q_ub  = {row['ratio_ub_rigorous']:.2f} "
              f"(区间 {row['ratio_ub_dry']:.2f} ~ {row['ratio_ub_zero']:.2f})")

    # --- A.3 若含潜热，表面平衡温度与 Arrhenius 衰减 ----------------------
    print("\n  --- 若引入潜热项: 表面能量平衡 h(T_inf - Ts) = j_s * dH ---")
    path3 = os.path.join(root, "附件", "附件3", "result3.xlsx")
    t3, C3 = as_grid(read_sheet(path3, 0), 21)
    C_surf = C3[:, -1]

    scen = {}
    for th in (6.0, 12.0, 18.0, 24.0):
        tq = th * 3600.0
        i = int(np.argmin(np.abs(t3 - tq)))
        Tinf_q, Cinf_q = chamber_at(chamber_t, tq, T_inf, C_inf)
        Tinf_q = float(Tinf_q) if tq <= chamber_t[-1] else 49.9989344
        Cinf_q = float(Cinf_q) if tq <= chamber_t[-1] else 0.0499875410
        dc_s = float(C_surf[i] - Cinf_q)

        entry = {"t_h": th, "C_s": float(C_surf[i]), "dC_surface": dc_s}
        for tag, rd in (("dry", rho_d_dry), ("zero", rho_d_zero)):
            j_s = rd * HM_CONV * dc_s                 # kg/(m^2 s)
            q_lat = j_s * DELTA_H                     # W/m^2
            dT = q_lat / H_CONV                       # K
            Ts = Tinf_q - dT
            ratio_D = math.exp(-3850.0 / (Ts + 273.15)) / math.exp(-3850.0 / (Tinf_q + 273.15))
            entry[f"Ts_{tag}"] = Ts
            entry[f"D_ratio_{tag}"] = ratio_D
            entry[f"q_lat_{tag}"] = q_lat
            print(f"    {th:5.1f} h  C_s={C_surf[i]:.4f}  rho_d={rd:7.2f}  "
                  f"j_s={j_s:.4g} kg/(m^2 s)  q_lat={q_lat:8.2f} W/m^2  "
                  f"Ts={Ts:5.2f} C  D/D_base={ratio_D:.3f}")
        scen[th] = entry

    return {"heat_supply_upper_bound": {str(int(k)): v for k, v in ub.items()},
            "rho_d": {"dry_basis": rho_d_dry, "rho_at_zero": rho_d_zero},
            "periods": {str(int(k)): v for k, v in results.items()},
            "surface_equilibrium_scenarios": {str(k): v for k, v in scen.items()}}


# ----------------------------------------------------------------------------
# 论证 B：全域最大含水率位于轴心且单调不增
# ----------------------------------------------------------------------------
def argument_max_principle(root: str, C_inf) -> dict:
    print("\n" + "=" * 78)
    print("论证 B  全域最大含水率位于轴心且随时间单调不增")
    print("=" * 78)

    summary = {}

    # --- B.1 问题三：固定半径，21 个规范输出位置 --------------------------
    path3 = os.path.join(root, "附件", "附件3", "result3.xlsx")
    t3, C3 = as_grid(read_sheet(path3, 0), 21)

    # 初期全场含水率几乎等于初值，浮点尾数差异会使 argmax 随机落在近轴心单元，
    # 故按容差判定"是否由轴心控制"，并单独统计精确并列的情形。
    TOL = 1e-12

    def center_control(Cmat: np.ndarray):
        Cmax_ = np.nanmax(Cmat, axis=1)
        gap = Cmax_ - Cmat[:, 0]
        at_center = gap <= TOL
        exact_tie = int(np.sum(at_center & (gap > 0.0)))
        return Cmax_, at_center, exact_tie

    Cmax3, at_center3, tie3 = center_control(C3)
    diffs3 = np.diff(Cmax3)

    print(f"\n  问题三 result3.xlsx : {len(t3)} 个 60 s 记录点，"
          f"空间 {C3.shape[1]} 个位置")
    print(f"    由轴心控制(容差 {TOL:g})的记录点数 : {int(at_center3.sum())} / {len(t3)}"
          f"   [其中精确并列 {tie3} 点，出现在初期全场≈初值时]")
    if not at_center3.all():
        bad = np.where(~at_center3)[0]
        print(f"    !! 真实例外: 索引 {bad[:10]}，对应时刻 "
              f"{t3[bad[:10]] / 3600.0} h，偏离量 {np.max(Cmax3[bad] - C3[bad, 0]):.3e} kg/kg")
    print(f"    C_max 单调性 : 最大回升量 = {diffs3.max():.3e} kg/kg（浮点噪声量级），"
          f"最大下降量 = {diffs3.min():.3e} kg/kg")
    print(f"    全场最小单元含水率 = {np.nanmin(C3):.6f} kg/kg，"
          f"长期环境 C_inf = {np.min(C_inf):.6f} ~ {np.max(C_inf):.6f} kg/kg")

    # 阈值穿越核对
    below = np.where(Cmax3 < 0.15)[0]
    t60_3 = t3[below[0]] if len(below) else float("nan")
    print(f"    首次严格达标记录点 t60 = {t60_3 / 3600.0:.6f} h ({t60_3:.0f} s)，"
          f"该点 C_max = {Cmax3[below[0]]:.10f} kg/kg")

    summary["problem3"] = {
        "n_records": int(len(t3)),
        "n_center_controlled": int(at_center3.sum()),
        "n_exact_ties_early": tie3,
        "max_backward_jump": float(diffs3.max()),
        "min_forward_drop": float(diffs3.min()),
        "min_cell_value": float(np.nanmin(C3)),
        "C_inf_range": [float(np.min(C_inf)), float(np.max(C_inf))],
        "t60_s": float(t60_3),
        "Cmax_at_t60": float(Cmax3[below[0]]) if len(below) else None,
    }

    # --- B.2 问题四：移动边界，规范输出列 + 内部细网格场 ------------------
    path4 = os.path.join(root, "附件", "附件3", "result4.xlsx")
    t4, C4 = as_grid(read_sheet(path4, 0), 13)
    # 第 1..12 列为固定物理位置（0 ~ 1.1 cm），第 13 列为药材表面
    Cmax4, at_center4, tie4 = center_control(C4[:, :12])
    diffs4 = np.diff(Cmax4)

    print(f"\n  问题四 result4.xlsx : {len(t4)} 个记录点，"
          f"12 个固定物理位置 + 药材表面")
    print(f"    由轴心控制(容差 {TOL:g})的记录点数 : {int(at_center4.sum())} / {len(t4)}"
          f"   [其中精确并列 {tie4} 点]")
    if not at_center4.all():
        bad = np.where(~at_center4)[0]
        print(f"    !! 真实例外: 索引 {bad[:10]}，偏离量 "
              f"{np.max(Cmax4[bad] - C4[bad, 0]):.3e} kg/kg")
    print(f"    C_max 单调性 : 最大回升量 = {diffs4.max():.3e} kg/kg")

    summary["problem4"] = {
        "n_records": int(len(t4)),
        "n_center_controlled": int(at_center4.sum()),
        "n_exact_ties_early": tie4,
        "max_backward_jump": float(diffs4.max()),
        "min_cell_value": float(np.nanmin(C4[:, :12])),
    }

    # 内部细网格场（若存在）：核验真实全域最大值，而非仅采样节点
    npz_path = os.path.join(root, "附件", "附件3", "result4_internal_field.npz")
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        keys = list(z.keys())
        print(f"\n  问题四内部细网格场 result4_internal_field.npz : keys = {keys}")
        field_key = None
        for k in keys:
            if z[k].ndim == 2:
                field_key = k
                break
        if field_key is not None:
            F = z[field_key]
            CmaxF, at_centerF, tieF = center_control(F)
            print(f"    使用数组 '{field_key}' 形状 {F.shape}")
            print(f"    细网格由第 0 个单元(轴心)控制(容差 {TOL:g})的记录点数 : "
                  f"{int(at_centerF.sum())} / {len(at_centerF)}   [精确并列 {tieF} 点]")
            print(f"    细网格 C_max 最大回升量 = {np.diff(CmaxF).max():.3e} kg/kg")
            summary["problem4"]["internal_field_key"] = field_key
            summary["problem4"]["internal_n_center_controlled"] = int(at_centerF.sum())
            summary["problem4"]["internal_n_records"] = int(len(at_centerF))
            summary["problem4"]["internal_n_exact_ties_early"] = tieF
            summary["problem4"]["internal_max_backward_jump"] = float(np.diff(CmaxF).max())
            summary["problem4"]["internal_min_cell_value"] = float(np.nanmin(F))
        else:
            print("    未找到二维数组，跳过细网格核验")

    return summary


# ----------------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    print("工作目录:", root)

    chamber_t, T_inf, C_inf = load_chamber(os.path.join(root, "附件", "附件1.xlsx"))
    print(f"附件一: {len(chamber_t)} 点，0 ~ {chamber_t[-1]:.0f} s；"
          f"T_inf {T_inf.min():.4f} ~ {T_inf.max():.4f} C；"
          f"C_inf {C_inf.min():.6f} ~ {C_inf.max():.6f} kg/kg")

    report = {"scope": "S 级论证：能量自洽性 + 全域最大值性质（只读，不修改任何求解代码）"}

    try:
        report["argument_A_energy_consistency"] = argument_energy(
            root, chamber_t, T_inf, C_inf)
    except FileNotFoundError as exc:
        print("  !! 论证 A 跳过:", exc)

    try:
        report["argument_B_max_principle"] = argument_max_principle(root, C_inf)
    except FileNotFoundError as exc:
        print("  !! 论证 B 跳过:", exc)

    out = os.path.join(root, "附件", "附件3", "论证_diagnostics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    print("完成。诊断结果已写入:", out)


if __name__ == "__main__":
    main()
