# -*- coding: utf-8 -*-
"""
多段レンズ系（DOE → f1,f2,...,fn）シミュレーション
v3: 各レンズ面での「2本のビーム間距離（gap）」を before/after で出力
"""
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Ray:
    y: float
    a: float

def M_free(L: float):
    return ((1.0, L),(0.0, 1.0))

def M_lens(f: float):
    return ((1.0, 0.0),(-1.0/f, 1.0))

def apply_M(M, r: Ray) -> Ray:
    A,B = M[0]; C,D = M[1]
    return Ray(A*r.y + B*r.a, C*r.y + D*r.a)

def propagate_to(z_from: float, z_to: float, r: Ray) -> Ray:
    return apply_M(M_free(z_to - z_from), r)

def _q_from_waist(lam_mm: float, w0_mm: float, z_from_waist_mm: float) -> complex:
    zR = math.pi*w0_mm**2/lam_mm
    return (z_from_waist_mm + 1j*zR)

def _q_from_wR(lam_mm: float, w_mm: float, R_mm: Optional[float]) -> complex:
    if (R_mm is None) or (math.isinf(R_mm)):
        inv_q = complex(0.0, -(lam_mm/(math.pi*w_mm**2)))
    else:
        inv_q = complex(1.0/R_mm, -(lam_mm/(math.pi*w_mm**2)))
    return 1.0/inv_q

def _w_from_q(lam_mm: float, q: complex) -> float:
    inv_q = 1.0/q
    if abs(inv_q.imag) < 1e-30: return float("inf")
    w2 = -(lam_mm/math.pi)/inv_q.imag
    return math.sqrt(max(0.0, w2))

def _propagate_q_free(q: complex, L: float) -> complex:
    A,B,C,D = 1.0, L, 0.0, 1.0
    return (A*q + B)/(C*q + D)

def _propagate_q_lens(q: complex, f: float) -> complex:
    A,B,C,D = 1.0, 0.0, -1.0/f, 1.0
    return (A*q + B)/(C*q + D)

def _waist_info_from_q(lam_mm: float, q_here: complex):
    zR = q_here.imag
    w0 = math.sqrt(lam_mm*zR/math.pi)
    s = -q_here.real
    return s, w0

def trace_system(wavelength_nm: float,
                 beam_diameter_mm: float,
                 doe_angle_deg: float,
                 lenses: List[Dict[str, float]],
                 w0_um: Optional[float]=None,
                 z0_mm: Optional[float]=None,
                 w_at_DOE_mm: Optional[float]=None,
                 R_at_DOE_mm: Optional[float]=None,
                 plot: bool=False,
                 refractive_index: float=1.0) -> Dict[str, object]:

    theta = math.radians(doe_angle_deg)
    ray_plus, ray_minus = Ray(0.0, +theta), Ray(0.0, -theta)

    z_curr = 0.0
    lam_mm = wavelength_nm*1e-6/refractive_index

    # Gaussian init
    gaussian_enabled = False
    if (w0_um is None and z0_mm is None and w_at_DOE_mm is None):
        if beam_diameter_mm and beam_diameter_mm>0:
            w_at_DOE_mm = beam_diameter_mm/2.0
            R_at_DOE_mm = float("inf")
    if (w0_um is not None and z0_mm is not None) or (w_at_DOE_mm is not None):
        gaussian_enabled = True
        if w0_um is not None and z0_mm is not None:
            q_here = _q_from_waist(lam_mm, w0_um*1e-3, (0.0 - z0_mm))
        else:
            q_here = _q_from_wR(lam_mm, w_at_DOE_mm, R_at_DOE_mm)
    else:
        q_here = None

    beam_gaps = []

    z_last_for_gauss = 0.0
    for i, lf in enumerate(lenses, start=1):
        zl, f = lf["z"], lf["f"]
        name = f"f{i}"

        # to lens
        ray_plus  = propagate_to(z_curr, zl, ray_plus)
        ray_minus = propagate_to(z_curr, zl, ray_minus)
        z_curr = zl
        gap_before = abs(ray_plus.y - ray_minus.y)

        if gaussian_enabled:
            q_here = _propagate_q_free(q_here, zl - z_last_for_gauss)
            z_last_for_gauss = zl

        # lens
        ray_plus  = apply_M(M_lens(f), ray_plus)
        ray_minus = apply_M(M_lens(f), ray_minus)
        gap_after = abs(ray_plus.y - ray_minus.y)
        beam_gaps.append({"name": name, "z_mm": zl, "gap_before_mm": gap_before, "gap_after_mm": gap_after})

        if gaussian_enabled:
            q_here = _propagate_q_lens(q_here, f)

    half_angle_after_last = abs(ray_plus.a)
    s_after_last = float("inf") if abs(ray_plus.a)<1e-12 else -ray_plus.y/ray_plus.a
    z_cross = z_curr + s_after_last

    pitch_mm = float("inf") if half_angle_after_last==0 else (wavelength_nm*1e-6/refractive_index)/(2*math.sin(half_angle_after_last))

    ray_diameter_last = abs(ray_plus.y - ray_minus.y)

    results = {
        "ray_results": {
            "half_angle_after_f4_deg": math.degrees(half_angle_after_last),
            "full_divergence_deg": 2*math.degrees(half_angle_after_last),
            "distance_after_f4_to_cross_mm": s_after_last,
            "z_cross_from_DOE_mm": z_cross,
            "ray_diameter_at_f4_mm": ray_diameter_last
        },
        "interference": {
            "fringe_pitch_mm": pitch_mm,
            "fringe_pitch_um": pitch_mm*1e3
        },
        "beam_gaps": beam_gaps
    }

    if gaussian_enabled:
        q_last = q_here
        s_waist_from_last, w0_after_last = _waist_info_from_q(lam_mm, q_last)
        if math.isfinite(s_after_last):
            q_at_cross = _propagate_q_free(q_last, s_after_last)
            w_cross = _w_from_q(lam_mm, q_at_cross)
        else:
            w_cross = float("inf")
        results["gaussian_results"] = {
            "diameter_at_f4_mm": 2*_w_from_q(lam_mm, q_last),
            "diameter_waist_mm": 2*w0_after_last,
            "diameter_at_cross_mm": 2*w_cross,
            "half_angle_farfield_deg": math.degrees(lam_mm/(math.pi*w0_after_last))
        }
    return results

# Demo run
if __name__ == "__main__":
    lenses = [
        {"z": 300.0, "f": 300.0},
        {"z": 660.0, "f": 60.0},
        {"z": 920.0, "f": 200.0},
        {"z":1062.0, "f":-60.0},  # f5 (凹)
        {"z":1085.0, "f":50.0},   # f6 (凸)
        {"z":1140.0, "f":20.0},
    ]
    res = trace_system(515.0, 5.0, 0.485, lenses)
    print("=== 各面ギャップ ===")
    for g in res["beam_gaps"]:
        print(f"{g['name']} (z={g['z_mm']} mm): before={g['gap_before_mm']:.3f} mm, after={g['gap_after_mm']:.3f} mm")
