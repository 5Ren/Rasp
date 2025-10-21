# -*- coding: utf-8 -*-
"""
4f系（DOE → f1,f2,f3,f4）を幾何光学でシミュレーション
- 入力:
  wavelength_nm: 使用波長 [nm]（幾何光学では直接は使わない・記録用）
  beam_diameter_mm: ビーム直径 [mm]（一次近似では未使用・将来拡張用）
  doe_angle_deg: DOEの回折角（±1次）[deg] ・・・DOE直後の光線角度を ±θ とする
  lenses: リスト[{"z": <DOEからの位置mm>, "f": <焦点距離mm>}, ...]   ※f1~f4順
  （ガウスビーム任意指定）
    - w0_um, z0_mm: ウエスト半径[µm]とDOE=0からの位置[mm]（ウエスト基準）
    - または w_at_DOE_mm, R_at_DOE_mm: DOE面のビーム半径[mm]と曲率半径R[mm]（R=∞可）

- 出力:
  ・Ray（幾何光学）:
      f4通過直後の2本の光線角（半角）[deg] と全発散角[deg]
      f4通過直後から見た交差点までの距離 s_after_f4 [mm]
      DOEから見た交差位置 z_cross [mm]
  ・Gaussian（q-parameter、任意）:
      w(f4)[mm], f4からのウエスト位置[mm], w0[mm], 遠方発散（λ/πw0）[deg]

前提:
  ・薄肉レンズ, パラキシャル近似（小角）: 光線ベクトル [y, α]（αはrad, 小角でtan~α）
  ・自由空間伝搬:  [[1, L],[0,1]]
  ・薄肉レンズ:    [[1, 0],[-1/f, 1]]
  ・2本の光線（+θと-θ）は対称なので、交差点は軸上に来る（理想系）。
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ========= 幾何光学（レイ） =========
@dataclass
class Ray:
    y: float      # height [mm]
    a: float      # angle [rad]

def M_free(L: float):
    """自由空間伝搬行列"""
    return ((1.0, L),
            (0.0, 1.0))

def M_lens(f: float):
    """薄肉レンズ行列"""
    return ((1.0, 0.0),
            (-1.0/f, 1.0))

def apply_M(M, r: Ray) -> Ray:
    A,B = M[0]
    C,D = M[1]
    y = A*r.y + B*r.a
    a = C*r.y + D*r.a
    return Ray(y,a)

def propagate_to(z_from: float, z_to: float, r: Ray) -> Ray:
    """現在位置 z_from から z_to へ自由空間伝搬"""
    L = z_to - z_from
    return apply_M(M_free(L), r)

# ========= ガウスビーム（q-parameter） =========
def _q_from_waist(wavelength_mm: float, w0_mm: float, z_from_waist_mm: float) -> complex:
    """ウエスト基準（waist at z=0）から、観測面までの z_from_waist_mm を持つq"""
    zR = math.pi * w0_mm**2 / wavelength_mm
    return (z_from_waist_mm + 1j*zR)

def _q_from_wR(wavelength_mm: float, w_mm: float, R_mm: Optional[float]) -> complex:
    """DOE面のビーム半径wと波面曲率半径Rからqを構成（1/q = 1/R - i λ/(π w^2)）"""
    if (R_mm is None) or (math.isinf(R_mm)):
        inv_q_im = -(wavelength_mm / (math.pi * w_mm**2))
        inv_q = complex(0.0, inv_q_im)
    else:
        inv_q = complex(1.0/R_mm, -(wavelength_mm / (math.pi * w_mm**2)))
    return 1.0 / inv_q

def _w_from_q(wavelength_mm: float, q: complex) -> float:
    """qからビーム半径wを返す（w^2 = - (λ/π) * Im(1/q)）"""
    inv_q = 1.0 / q
    w2 = - (wavelength_mm / math.pi) * inv_q.imag
    if w2 < 0:
        w2 = 0.0
    return math.sqrt(w2)

def _propagate_q_free(q: complex, L: float) -> complex:
    """自由空間伝搬（ABCD: [[1,L],[0,1]]）"""
    A,B,C,D = 1.0, L, 0.0, 1.0
    return (A*q + B) / (C*q + D)

def _propagate_q_lens(q: complex, f: float) -> complex:
    """薄肉レンズ通過（ABCD: [[1,0],[-1/f,1]]）"""
    A,B,C,D = 1.0, 0.0, -1.0/f, 1.0
    return (A*q + B) / (C*q + D)

def _waist_info_from_q(wavelength_mm: float, q_here: complex):
    """観測面のqから、そこからのウエスト位置とw0を返す"""
    zR = q_here.imag
    w0 = math.sqrt(wavelength_mm * zR / math.pi)
    s_from_here_to_waist = - q_here.real   # 観測面からwaistまでの距離（+なら下流）
    return s_from_here_to_waist, w0

# ========= メイン：レイ＋（任意で）ガウス =========
def trace_system(wavelength_nm: float,
                 beam_diameter_mm: float,
                 doe_angle_deg: float,
                 lenses: List[Dict[str, float]],
                 # ---- optional Gaussian ----
                 w0_um: Optional[float] = None,
                 z0_mm: Optional[float] = None,
                 w_at_DOE_mm: Optional[float] = None,
                 R_at_DOE_mm: Optional[float] = None,
                 plot: bool = False
                 ) -> Dict[str, object]:
    """
    Rayは常に計算。Gaussianは初期条件が与えられた場合のみ計算。
    """
    # ---- レイ初期化（DOE=0、±θ）
    theta = math.radians(doe_angle_deg)
    ray_plus  = Ray(y=0.0, a=+theta)
    ray_minus = Ray(y=0.0, a=-theta)

    z_curr = 0.0
    history_plus  = [{"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what":"DOE"}]
    history_minus = [{"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what":"DOE"}]

    # ---- ガウス初期化（必要なら）
    gaussian_enabled = False
    wavelength_mm = wavelength_nm * 1e-6
    w_profile = []   # (z, w)
    if (w0_um is not None and z0_mm is not None) or (w_at_DOE_mm is not None):
        gaussian_enabled = True
        if w0_um is not None and z0_mm is not None:
            q_here = _q_from_waist(wavelength_mm, w0_mm=w0_um*1e-3, z_from_waist_mm=(0.0 - z0_mm))
        else:
            q_here = _q_from_wR(wavelength_mm, w_mm=w_at_DOE_mm, R_mm=R_at_DOE_mm)
        w_profile.append((0.0, _w_from_q(wavelength_mm, q_here)))
    else:
        q_here = None  # type: ignore

    # ---- 各レンズへ順に伝搬→通過
    z_f4 = lenses[-1]["z"] if lenses else 0.0
    for i, lf in enumerate(lenses, start=1):
        zl = lf["z"]
        f  = lf["f"]

        # 自由空間
        ray_plus  = propagate_to(z_curr, zl, ray_plus)
        ray_minus = propagate_to(z_curr, zl, ray_minus)
        z_curr = zl
        history_plus.append({"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what": f"before f{i}"})
        history_minus.append({"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what": f"before f{i}"})

        if gaussian_enabled:
            q_here = _propagate_q_free(q_here, zl - (w_profile[-1][0]))  # move from previous z to zl
            w_profile.append((zl, _w_from_q(wavelength_mm, q_here)))

        # レンズ通過
        ray_plus  = apply_M(M_lens(f), ray_plus)
        ray_minus = apply_M(M_lens(f), ray_minus)
        history_plus.append({"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what": f"after f{i}"})
        history_minus.append({"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what": f"after f{i}"})

        if gaussian_enabled:
            q_here = _propagate_q_lens(q_here, f)
            w_profile.append((zl, _w_from_q(wavelength_mm, q_here)))

    # ---- f4通過直後の発散角（レイ）
    half_angle_after_f4_deg = abs(math.degrees(ray_plus.a))
    full_divergence_deg = 2.0 * half_angle_after_f4_deg

    # ---- f4直後からの交差距離と位置（レイ）
    if abs(ray_plus.a) < 1e-12:
        s_after_f4 = math.inf
    else:
        s_after_f4 = - ray_plus.y / ray_plus.a
    z_cross = z_curr + s_after_f4

    results = {
        "inputs": {
            "wavelength_nm": wavelength_nm,
            "beam_diameter_mm": beam_diameter_mm,
            "doe_angle_deg": doe_angle_deg,
            "lenses": lenses
        },
        "ray_results": {
            "half_angle_after_f4_deg": half_angle_after_f4_deg,
            "full_divergence_deg": full_divergence_deg,
            "distance_after_f4_to_cross_mm": s_after_f4,
            "z_cross_from_DOE_mm": z_cross
        },
        "history_plus": history_plus,
        "history_minus": history_minus
    }

    # ---- ガウスの集計（任意）
    if gaussian_enabled:
        # f4面のqからウエスト位置などを算出
        # q_here は f4通過直後のq
        s_waist_from_f4, w0_after_f4_mm = _waist_info_from_q(wavelength_mm, q_here)
        z_waist_from_DOE = z_f4 + s_waist_from_f4
        # 遠方発散（半角）
        half_angle_gauss_rad = wavelength_mm / (math.pi * w0_after_f4_mm)
        half_angle_gauss_deg = math.degrees(half_angle_gauss_rad)
        full_angle_gauss_deg = 2.0 * half_angle_gauss_deg

        results["gaussian_results"] = {
            "w_at_f4_mm": _w_from_q(wavelength_mm, q_here),
            "waist_from_f4_mm": s_waist_from_f4,
            "waist_from_DOE_mm": z_waist_from_DOE,
            "w0_after_f4_mm": w0_after_f4_mm,
            "half_angle_farfield_deg": half_angle_gauss_deg,
            "full_angle_farfield_deg": full_angle_gauss_deg,
            "w_profile": w_profile  # [(z, w)]
        }

        # プロット（任意）
        if plot:
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

# ===== 使い方例 =====
if __name__ == "__main__":
    # 図のように DOE→f1→f2→f3→f4 の順で位置[z]と焦点距離[f]を与える
    wavelength_nm = 515.0
    beam_diameter_mm = 5.0
    doe_angle_deg = 0.485

    lenses = [
        {"z": 300.0, "f": 300.0},  # f1 at z=300 mm, f=300 mm
        {"z": 660.0, "f": 60.0},   # f2 at z=660 mm, f=60 mm
        {"z": 920.0, "f": 200.0},  # f3 at z=920 mm, f=200 mm
        {"z": 1140.0,"f": 20.0},   # f4 at z=1140 mm, f=20 mm
    ]

    # --- ガウス初期条件の例（DOE面w=1.1mm, R=∞の準平行光） ---
    res = trace_system(wavelength_nm, beam_diameter_mm, doe_angle_deg, lenses,
                       w_at_DOE_mm=1.1, R_at_DOE_mm=float("inf"),
                       plot=True)

    from pprint import pprint
    print("=== RAY RESULTS ===")
    pprint(res["ray_results"])
    if "gaussian_results" in res:
        print("\\n=== GAUSSIAN RESULTS ===")
        tmp = dict(res["gaussian_results"])
        tmp.pop("w_profile", None)  # 省略表示
        pprint(tmp)
