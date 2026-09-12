# -*- coding: utf-8 -*-
"""
A 级论证之二：附件二收缩数据与模型含水率场的自洽性检验

问题四把附件二的实测半径 R(t) 作为外生输入，含水率 C 由扩散方程独立算出。
本脚本检验两者能否用"水分驱动的理想收缩律"统一起来。

理想收缩律推导（长度不变、仅径向收缩、无孔隙）：
    干物质守恒      ->  rho_d(t) * R(t)^2 = const
    体积 = 干物质体积 + 水分体积
    令 phi = rho_w / rho_s（等效干物质体积比），则
        (R/R0)^2 = (phi + C) / (phi + C0)
    反解
        phi(t) = (s*C0 - C) / (1 - s),    s = (R/R0)^2

关键点：该式中的 C 是**驱动收缩的那个含水率**。问题四的"均匀比例收缩"假设意味着
C 取体积平均值 Cbar；但半径本身是表面量，物理上更可能由外层含水率控制。
本脚本对三种候选驱动量分别反解 phi，比较哪一个能让 phi 沿全程保持恒定。

运行：python 论证_收缩与含水率自洽性.py
输出：终端报表 + 附件/附件3/论证_收缩自洽性.json
"""

from __future__ import annotations

import json
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

R0_CM = 2.000
C0 = 2.55
RHO_W = 1000.0


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


def as_grid(rows):
    t_list, data = [], []
    for r in rows[1:]:
        if not r or r[0] is None:
            continue
        try:
            tt = float(r[0])
        except (TypeError, ValueError):
            continue
        vals = []
        for j in range(1, len(r)):
            raw = r[j]
            try:
                vals.append(float(raw))
            except (TypeError, ValueError):
                vals.append(np.nan)
        t_list.append(tt)
        data.append(vals)
    return np.asarray(t_list, float), np.asarray(data, float)


def least_squares_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(coef[1]), (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def evaluate_driver(name: str, C_driver: np.ndarray, s: np.ndarray, R_m: np.ndarray,
                    t_s: np.ndarray):
    """对给定驱动量反解 phi 并评估理想收缩律的恒定性。"""
    denom = 1.0 - s
    valid = (s < 0.90) & (C_driver < 0.98 * C0) & (denom > 1e-6)
    phi = np.full_like(C_driver, np.nan)
    phi[valid] = (s[valid] * C0 - C_driver[valid]) / denom[valid]

    idx = np.where(valid)[0]
    segs = np.array_split(idx, 4)
    seg_mean = [float(np.nanmean(phi[sg])) for sg in segs]

    # 固定时间窗口，用于区分初期异常段与后期稳定段
    windows = {}
    for lo, hi in ((0.0, 6.0), (6.0, 12.0), (12.0, 24.0), (24.0, 40.0), (40.0, 1e9)):
        m = valid & (t_s >= lo * 3600.0) & (t_s < (hi if hi < 1e8 else 1e18) * 3600.0)
        if m.sum() < 5:
            windows[f"{lo:g}-{hi:g}h"] = None
            continue
        windows[f"{lo:g}-{hi:g}h"] = {
            "n": int(m.sum()),
            "phi_mean": float(np.nanmean(phi[m])),
            "phi_std": float(np.nanstd(phi[m])),
            "phi_min": float(np.nanmin(phi[m])),
            "phi_max": float(np.nanmax(phi[m])),
        }

    phi_mean = float(np.nanmean(phi))
    phi_std = float(np.nanstd(phi))
    cv = phi_std / abs(phi_mean) * 100 if phi_mean != 0 else float("inf")

    # 用平均 phi 回代，评估理想收缩律对半径的复现能力
    s_fit = (phi_mean + C_driver) / (phi_mean + C0)
    R_fit = np.sqrt(np.maximum(s_fit, 0.0)) * (R0_CM / 100.0)
    resid = (R_fit - R_m) * 100.0          # cm
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((R_m * 100.0 - (R_m * 100.0).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "name": name,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "phi_cv_percent": cv,
        "phi_min": float(np.nanmin(phi)),
        "phi_max": float(np.nanmax(phi)),
        "has_negative_phi": bool(np.nanmin(phi) < 0),
        "segment_phi_means": seg_mean,
        "segment_drift": float(max(seg_mean) - min(seg_mean)),
        "windows": windows,
        "equivalent_solid_density_kg_m3": float(RHO_W / phi_mean) if phi_mean > 0 else None,
        "radius_residual_max_cm": float(np.abs(resid).max()),
        "radius_residual_rms_cm": float(np.sqrt(np.mean(resid ** 2))),
        "radius_residual_max_rel_percent": float(
            np.max(np.abs(resid) / (R_m * 100.0)) * 100),
        "ideal_law_r2": r2,
    }


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    print("工作目录:", root)

    # ---------------------------------------------------------- 附件二 R(t)
    t_R, R_all = as_grid(read_sheet(os.path.join(root, "附件", "附件2.xlsx"), 0))
    R_cm = R_all[:, 0]
    print(f"\n附件二: {len(t_R)} 点，{t_R[0] / 3600:.0f} ~ {t_R[-1] / 3600:.0f} h，"
          f"R = {R_cm.min():.4f} ~ {R_cm.max():.4f} cm，"
          f"总径向收缩率 = {(R0_CM - R_cm[-1]) / R0_CM * 100:.1f}%")

    # ---------------------------------------- 问题四内部细网格场
    z = np.load(os.path.join(root, "附件", "附件3", "result4_internal_field.npz"))
    t_s = z["times_s"].astype(float)
    xi = z["xi_centers"].astype(float)
    C_field = z["moisture"].astype(float)
    R_m = z["surface_radii_m"].astype(float)
    print(f"内部细网格: {len(t_s)} 时间层 x {C_field.shape[1]} 单元，至 {t_s[-1] / 3600:.4f} h")

    # 参考域体积加权平均（权重 xi dxi）
    Cbar = 2.0 * np.trapz(C_field * xi[None, :], xi, axis=1)

    # 外层半域平均（xi in [0.5, 1]），反映近表面区域的整体含水量
    m_out = xi >= 0.5
    w_out = xi[m_out]
    Cbar_out = np.trapz(C_field[:, m_out] * w_out[None, :], w_out, axis=1) / np.trapz(w_out, w_out)

    # 表面重构值：直接取 result4.xlsx 的"药材表面"列（问题四的物理解析重构结果）
    t_r4, C4 = as_grid(read_sheet(os.path.join(root, "附件", "附件3", "result4.xlsx"), 0))
    C_surf_r4 = C4[:, -1]
    C_surf = np.interp(t_s, t_r4, C_surf_r4)

    print(f"体积平均含水率 Cbar      : {Cbar[0]:.4f} -> {Cbar[-1]:.4f} kg/kg")
    print(f"外层半域平均 Cbar_outer   : {Cbar_out[0]:.4f} -> {Cbar_out[-1]:.4f} kg/kg")
    print(f"表面重构值   C_surface    : {C_surf[0]:.4f} -> {C_surf[-1]:.4f} kg/kg")

    # --------------------- [0] 输入一致性：模型是否按附件二插值了半径
    R_excel_m = np.interp(t_s, t_R, R_cm) / 100.0
    dev = float(np.abs(R_excel_m - R_m).max())
    print(f"\n[0] 模型半径输入核对: 与附件二分段线性插值的最大偏差 = {dev:.3e} m "
          f"({dev * 1e5:.3e} cm)")
    assert dev < 1e-9, "模型半径与附件二插值不一致，后续检验无意义"

    # --------------------- [1] 三种驱动量的理想收缩律恒定性比较
    s = (R_m / (R0_CM / 100.0)) ** 2
    drivers = [("体积平均 Cbar", Cbar),
               ("外层半域平均 Cbar_outer", Cbar_out),
               ("表面重构值 C_surface", C_surf)]

    print("\n" + "=" * 78)
    print("[1] 理想收缩律 (R/R0)^2 = (phi + C)/(phi + C0) 的驱动量判别")
    print("=" * 78)

    results = []
    for name, C_drv in drivers:
        r = evaluate_driver(name, C_drv, s, R_m, t_s)
        results.append(r)
        print(f"\n  驱动量: {name}")
        print(f"    全程: phi 均值 = {r['phi_mean']:9.5f}  标准差 = {r['phi_std']:.3e}  "
              f"变异系数 = {r['phi_cv_percent']:8.3f}%")
        print(f"    phi 范围 = [{r['phi_min']:.5f}, {r['phi_max']:.5f}]"
              f"{'   <-- 出现负值，物理上不可能' if r['has_negative_phi'] else ''}")
        print(f"    分窗口 phi:")
        for wname, w in r["windows"].items():
            if w is None:
                print(f"      {wname:12s}  （样本过少）")
            else:
                print(f"      {wname:12s} n={w['n']:5d}   "
                      f"phi = {w['phi_mean']:8.4f} +- {w['phi_std']:8.4f}   "
                      f"范围 [{w['phi_min']:8.4f}, {w['phi_max']:8.4f}]")
        if r["equivalent_solid_density_kg_m3"]:
            print(f"    全程均值对应 rho_s = {r['equivalent_solid_density_kg_m3']:.1f} kg/m^3"
                  f"   （植物物料常见范围约 1200~1600）")
        print(f"    用全程均值 phi 回代: 半径最大残差 = {r['radius_residual_max_cm']:.4e} cm "
              f"({r['radius_residual_max_rel_percent']:.3f}%),  R^2 = {r['ideal_law_r2']:.6f}")

    best = min(results, key=lambda r: (r["has_negative_phi"], r["phi_cv_percent"]))
    print(f"\n  -> phi 最恒定（变异系数最小且无负值）的驱动量: {best['name']}")
    print(f"     变异系数 {best['phi_cv_percent']:.3f}% vs "
          f"体积平均口径的 {results[0]['phi_cv_percent']:.3f}%")

    # --------------------- [2] 与常用收缩模型的拟合优度比较
    print("\n" + "=" * 78)
    print("[2] 与常用收缩模型的形式比较（以各自的驱动量为自变量）")
    print("=" * 78)
    a1, b1, r2_lin = least_squares_fit(Cbar / C0, R_m * 100.0 / R0_CM)
    a2, b2, r2_sq = least_squares_fit(Cbar / C0, (R_m * 100.0 / R0_CM) ** 2)
    n_pow, _, r2_pow = least_squares_fit(np.log(Cbar / C0), np.log(R_m * 100.0 / R0_CM))
    print(f"  (a) 线性    R/R0 = {a1:.5f}*(Cbar/C0) + {b1:.5f}          R^2 = {r2_lin:.6f}")
    print(f"  (b) 体积线性 V/V0 = {a2:.5f}*(Cbar/C0) + {b2:.5f}          R^2 = {r2_sq:.6f}")
    print(f"  (c) 幂律    R/R0 = (Cbar/C0)^{n_pow:.5f}                 R^2 = {r2_pow:.6f}")
    print("  注: R(t) 与任何含水率指标均在时间上单调，故这些单调模型都能得到高 R^2，")
    print("      R^2 本身不具判别力；判别力来自 [1] 中 phi 的恒定性检验。")

    # --------------------- [3] 时间同步性
    print("\n" + "=" * 78)
    print("[3] 时间同步性  dR/dC （相对失水速率）")
    print("=" * 78)
    sync = {}
    for name, C_drv in drivers:
        dR = np.gradient(R_m, t_s)
        dC = np.gradient(C_drv, t_s)
        ok = np.abs(dC) > 1e-12
        ratio = dR[ok] / dC[ok]
        cv = ratio.std() / abs(ratio.mean()) * 100
        sync[name] = {"mean": float(ratio.mean()), "std": float(ratio.std()),
                      "cv_percent": float(cv)}
        print(f"  {name:26s} 均值 = {ratio.mean():.5e}  变异系数 = {cv:8.2f}%")

    # ------------------------------------------------------- 结果落盘
    report = {
        "scope": "附件二实测收缩与问题四模型含水率场的自洽性检验（只读，不修改求解代码）",
        "ideal_shrinkage_law": "(R/R0)^2 = (phi + C_driver) / (phi + C0),  phi = rho_w / rho_s",
        "input_check_max_radius_deviation_m": dev,
        "moisture_measures": {
            "Cbar_volume_average": [float(Cbar[0]), float(Cbar[-1])],
            "Cbar_outer_half": [float(Cbar_out[0]), float(Cbar_out[-1])],
            "C_surface_reconstructed": [float(C_surf[0]), float(C_surf[-1])],
        },
        "driver_comparison": results,
        "best_driver": best["name"],
        "classical_model_fits": {
            "linear_R_over_R0": {"slope": a1, "intercept": b1, "r2": r2_lin},
            "linear_V_over_V0": {"slope": a2, "intercept": b2, "r2": r2_sq},
            "power_law": {"exponent": n_pow, "r2": r2_pow},
        },
        "time_synchrony": sync,
    }
    out = os.path.join(root, "附件", "附件3", "论证_收缩自洽性.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    print("完成。结果已写入:", out)


if __name__ == "__main__":
    main()
