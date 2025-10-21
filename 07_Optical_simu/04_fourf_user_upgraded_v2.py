# -*- coding: utf-8 -*-
"""
4f系（DOE → f1,f2,f3,f4）を幾何光学でシミュレーション（レイ＋任意でガウス）

更新点:
- 交差位置でのスポット直径 `spot_diameter_at_cross_mm` を常に出力
  - ガウスあり: 2*w(z_cross)
  - ガウスなし: Airy径の推定 d ≈ 2.44*(λ/n)*f4 / D （Dはf4面ビーム径; Rayから計算）
- 既存の干渉ピッチ・ビーム径出力は維持

注: beam_diameter_mm は「DOE入射ビーム直径」。ガウス初期条件が未指定なら w(DOE)=beam_diameter_mm/2, R=∞で自動設定。
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from numpy.f2py.common_rules import f2py_version


# ========= 幾何光学（レイ） =========
@dataclass
class Ray:
    y: float      # height [mm]
    a: float      # angle [rad]

def M_free(L: float):
    return ((1.0, L),
            (0.0, 1.0))

def M_lens(f: float):
    return ((1.0, 0.0),
            (-1.0/f, 1.0))

def apply_M(M, r: Ray) -> Ray:
    A,B = M[0]
    C,D = M[1]
    y = A*r.y + B*r.a
    a = C*r.y + D*r.a
    return Ray(y,a)

def propagate_to(z_from: float, z_to: float, r: Ray) -> Ray:
    L = z_to - z_from
    return apply_M(M_free(L), r)

# ========= ガウスビーム（q-parameter） =========
def _q_from_waist(wavelength_mm: float, w0_mm: float, z_from_waist_mm: float) -> complex:
    zR = math.pi * w0_mm**2 / wavelength_mm
    return (z_from_waist_mm + 1j*zR)

def _q_from_wR(wavelength_mm: float, w_mm: float, R_mm: Optional[float]) -> complex:
    if (R_mm is None) or (math.isinf(R_mm)):
        inv_q_im = -(wavelength_mm / (math.pi * w_mm**2))
        inv_q = complex(0.0, inv_q_im)
    else:
        inv_q = complex(1.0/R_mm, -(wavelength_mm / (math.pi * w_mm**2)))
    return 1.0 / inv_q

def _w_from_q(wavelength_mm: float, q: complex) -> float:
    """w^2 = - (λ/π) / Im(1/q)"""
    inv_q = 1.0 / q
    if abs(inv_q.imag) < 1e-30:
        return float("inf")
    w2 = - (wavelength_mm / math.pi) / inv_q.imag
    if w2 < 0:
        w2 = 0.0
    return math.sqrt(w2)

def _propagate_q_free(q: complex, L: float) -> complex:
    A,B,C,D = 1.0, L, 0.0, 1.0
    return (A*q + B) / (C*q + D)

def _propagate_q_lens(q: complex, f: float) -> complex:
    A,B,C,D = 1.0, 0.0, -1.0/f, 1.0
    return (A*q + B) / (C*q + D)

def _waist_info_from_q(wavelength_mm: float, q_here: complex):
    zR = q_here.imag
    w0 = math.sqrt(wavelength_mm * zR / math.pi)
    s_from_here_to_waist = - q_here.real
    return s_from_here_to_waist, w0

# ========= メイン =========
def trace_system(wavelength_nm: float,
                 beam_diameter_mm: float,
                 doe_angle_deg: float,
                 lenses: List[Dict[str, float]],
                 # optional Gaussian
                 w0_um: Optional[float] = None,
                 z0_mm: Optional[float] = None,
                 w_at_DOE_mm: Optional[float] = None,
                 R_at_DOE_mm: Optional[float] = None,
                 plot: bool = False,
                 refractive_index: float = 1.0) -> Dict[str, object]:

    # Rays at DOE
    theta = math.radians(doe_angle_deg)
    ray_plus  = Ray(y=0.0, a=+theta)
    ray_minus = Ray(y=0.0, a=-theta)

    z_curr = 0.0
    history_plus  = [{"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what":"DOE"}]
    history_minus = [{"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what":"DOE"}]

    lam_vac_mm = wavelength_nm * 1e-6
    lam_eff_mm = lam_vac_mm / refractive_index

    # If Gaussian not specified, use beam_diameter_mm at DOE (radius = diameter/2), R=inf
    gaussian_enabled = False
    if (w0_um is None and z0_mm is None and w_at_DOE_mm is None):
        if beam_diameter_mm and beam_diameter_mm > 0:
            w_at_DOE_mm = beam_diameter_mm / 2.0
            R_at_DOE_mm = float("inf")

    w_profile = []
    if (w0_um is not None and z0_mm is not None) or (w_at_DOE_mm is not None):
        gaussian_enabled = True
        if w0_um is not None and z0_mm is not None:
            q_here = _q_from_waist(lam_eff_mm, w0_mm=w0_um*1e-3, z_from_waist_mm=(0.0 - z0_mm))
        else:
            q_here = _q_from_wR(lam_eff_mm, w_mm=w_at_DOE_mm, R_mm=R_at_DOE_mm)
        w_profile.append((0.0, _w_from_q(lam_eff_mm, q_here)))
        w_at_DOE_for_report = _w_from_q(lam_eff_mm, q_here)
    else:
        q_here = None
        w_at_DOE_for_report = None

    # propagate to each lens
    z_f4 = lenses[-1]["z"] if lenses else 0.0
    last_z_for_gauss = 0.0
    for i, lf in enumerate(lenses, start=1):
        zl, f = lf["z"], lf["f"]
        ray_plus  = propagate_to(z_curr, zl, ray_plus)
        ray_minus = propagate_to(z_curr, zl, ray_minus)
        z_curr = zl
        history_plus.append({"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what": f"before f{i}"})
        history_minus.append({"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what": f"before f{i}"})

        if gaussian_enabled:
            q_here = _propagate_q_free(q_here, zl - last_z_for_gauss)
            last_z_for_gauss = zl
            w_profile.append((zl, _w_from_q(lam_eff_mm, q_here)))

        ray_plus  = apply_M(M_lens(f), ray_plus)
        ray_minus = apply_M(M_lens(f), ray_minus)
        history_plus.append({"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what": f"after f{i}"})
        history_minus.append({"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what": f"after f{i}"})

        if gaussian_enabled:
            q_here = _propagate_q_lens(q_here, f)
            w_profile.append((zl, _w_from_q(lam_eff_mm, q_here)))

    # angles and crossing
    half_angle_after_f4_rad = abs(ray_plus.a)
    half_angle_after_f4_deg = math.degrees(half_angle_after_f4_rad)
    full_divergence_deg = 2.0 * half_angle_after_f4_deg

    s_after_f4 = math.inf if abs(ray_plus.a) < 1e-12 else - ray_plus.y / ray_plus.a
    z_cross = z_curr + s_after_f4

    # Interference pitch (two-beam)
    pitch_mm = math.inf if half_angle_after_f4_rad == 0 else lam_eff_mm / (2.0 * math.sin(half_angle_after_f4_rad))

    # Beam diameter at f4 (ray-based)
    ray_diameter_at_f4_mm = 2.0 * abs(history_plus[-1]["y"])

    # Assemble results
    results = {
        "inputs": {
            "wavelength_nm": wavelength_nm,
            "refractive_index": refractive_index,
            "beam_diameter_mm": beam_diameter_mm,
            "doe_angle_deg": doe_angle_deg,
            "lenses": lenses
        },
        "ray_results": {
            "half_angle_after_f4_deg": half_angle_after_f4_deg,
            "full_divergence_deg": full_divergence_deg,
            "distance_after_f4_to_cross_mm": s_after_f4,
            "z_cross_from_DOE_mm": z_cross,
            "ray_diameter_at_f4_mm": ray_diameter_at_f4_mm
        },
        "interference": {
            "fringe_pitch_mm": pitch_mm,
            "fringe_pitch_um": pitch_mm * 1e3
        },
        "history_plus": history_plus,
        "history_minus": history_minus
    }

    # Spot diameter at crossing:
    # prefer Gaussian, otherwise Airy estimate using f4 and D at f4
    f4_focal = lenses[-1]["f"] if lenses else None
    if gaussian_enabled:
        q_f4 = q_here
        s_waist_from_f4, w0_after_f4_mm = _waist_info_from_q(lam_eff_mm, q_f4)
        z_waist_from_DOE = z_f4 + s_waist_from_f4

        if math.isfinite(s_after_f4):
            q_at_cross = _propagate_q_free(q_f4, s_after_f4)
            w_at_cross_mm = _w_from_q(lam_eff_mm, q_at_cross)
        else:
            w_at_cross_mm = float("inf")

        half_angle_gauss_rad = lam_eff_mm / (math.pi * w0_after_f4_mm)
        half_angle_gauss_deg = math.degrees(half_angle_gauss_rad)

        results["gaussian_results"] = {
            "w_at_DOE_mm": w_at_DOE_for_report,
            "diameter_at_DOE_mm": (2*w_at_DOE_for_report) if (w_at_DOE_for_report is not None) else None,
            "w_at_f4_mm": _w_from_q(lam_eff_mm, q_f4),
            "diameter_at_f4_mm": 2.0 * _w_from_q(lam_eff_mm, q_f4),
            "waist_from_f4_mm": s_waist_from_f4,
            "waist_from_DOE_mm": z_waist_from_DOE,
            "w0_after_f4_mm": w0_after_f4_mm,
            "diameter_waist_mm": 2.0 * w0_after_f4_mm,
            "w_at_cross_mm": w_at_cross_mm,
            "diameter_at_cross_mm": 2.0 * w_at_cross_mm,
            "half_angle_farfield_deg": half_angle_gauss_deg,
            "full_angle_farfield_deg": 2.0 * half_angle_gauss_deg
        }

        results["spot_diameter_at_cross_mm"] = 2.0 * w_at_cross_mm
        results["spot_method"] = "gaussian"
    else:
        # Airy estimate at the focus/crossing
        if f4_focal is not None and ray_diameter_at_f4_mm > 0:
            airy_mm = 2.44 * lam_eff_mm * f4_focal / ray_diameter_at_f4_mm
        else:
            airy_mm = float("nan")
        results["spot_diameter_at_cross_mm"] = airy_mm
        results["spot_method"] = "airy_estimate"

    # Optional plot
    if plot and gaussian_enabled:
        try:
            import matplotlib.pyplot as plt
            z_vals = [zw[0] for zw in w_profile]
            w_vals_um = [zw[1]*1e3 for zw in w_profile]  # mm -> µm
            plt.figure()
            plt.plot(z_vals, w_vals_um)
            for lf in lenses:
                plt.axvline(lf["z"], linestyle="--")
            if math.isfinite(z_cross):
                plt.axvline(z_cross, linestyle=":")
            plt.xlabel("z from DOE [mm]")
            plt.ylabel("Beam radius w(z) [µm]")
            plt.title("Gaussian beam radius along z")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            results["plot_error"] = str(e)

    return results


# ===== 簡単な動作確認 =====
if __name__ == "__main__":
    wavelength_nm = 515.0
    beam_diameter_mm = 5.0
    doe_angle_deg = 0.485
    lenses = [
        {"z": 300.0, "f": 300.0},
        {"z": 660.0, "f": 60.0},
        {"z": 920.0, "f": 200.0},
        {"z":1062.0, "f":-60.0},  # f5 (凹)
        {"z":1085.0, "f":50.0},   # f6 (凸)
        # {"z": 1002.0, "f": 200.0},
        {"z": 1140.0,"f": 20.0},
    ]
    res = trace_system(wavelength_nm, beam_diameter_mm, doe_angle_deg, lenses, plot=False)
    from pprint import pprint
    print("=== spot @ cross ===", res["spot_diameter_at_cross_mm"], "mm | method:", res["spot_method"])
    print("=== interference ==="); pprint(res["interference"])
    print("=== ray ==="); pprint(res["ray_results"])
    if "gaussian_results" in res: print("=== gaussian ==="); pprint(res["gaussian_results"])

    # ==== 日本語で結果表示 ====
    print("=== 光学系シミュレーション結果 ===")

    # --- 幾何光学（レイ）情報 ---
    print(f"f4でのビーム間距離: {res['ray_results']['ray_diameter_at_f4_mm']:.3f} mm")
    print(f"干渉入射角（±）: {res['ray_results']['half_angle_after_f4_deg']:.3f} °")
    print(f"f4通過後，交差位置: {res['ray_results']['z_cross_from_DOE_mm']:.3f} mm")
    print(f"f4通過後，交差位置（f4からの距離）: {res['ray_results']['distance_after_f4_to_cross_mm']:.3f} mm")

    # --- 干渉情報 ---
    print(f"\n干渉ピッチ: {res['interference']['fringe_pitch_um']:.3f} µm")

    # --- ガウスビーム情報 ---
    if "gaussian_results" in res:
        g = res["gaussian_results"]
        print(f"\nf4でのビーム直径: {g['diameter_at_f4_mm']:.4f} mm")
        print(f"ビームウェスト（最小径）: {g['diameter_waist_mm']:.4f} mm")
        print(f"交差位置でのビーム直径: {g['diameter_at_cross_mm']:.4f} mm")
        print(f"\n遠方発散角（±1/e²定義）: {g['half_angle_farfield_deg']:.4f} °")
    else:
        print("\nガウスビーム未指定（幾何光学のみ）")

    print("\n==========================")
