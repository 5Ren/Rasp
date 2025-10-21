# -*- coding: utf-8 -*-
"""
4f系（DOE → f1,f2,f3,f4）を幾何光学でシミュレーション
- 入力:
  wavelength_nm: 使用波長 [nm]（幾何光学では直接は使わない・記録用）
  beam_diameter_mm: ビーム直径 [mm]（一次近似では未使用・将来拡張用）
  doe_angle_deg: DOEの回折角（±1次）[deg] ・・・DOE直後の光線角度を ±θ とする
  lenses: リスト[{"z": <DOEからの位置mm>, "f": <焦点距離mm>}, ...]   ※f1~f4順
- 出力:
  ・f4通過直後の2本の光線角（半角）[deg] と全発散角[deg]
  ・f4通過直後から見た交差点までの距離 s_after_f4 [mm]
  ・DOEから見た交差位置 z_cross [mm]
  ・デバッグ用：各面通過時の (z, y, angle) 履歴

前提:
  ・薄肉レンズ, パラキシャル近似（小角）: 光線ベクトル [y, α]（αはrad, 小角でtan~α）
  ・自由空間伝搬:  [[1, L],[0,1]]
  ・薄肉レンズ:    [[1, 0],[-1/f, 1]]
  ・2本の光線（+θと-θ）は対称なので、交差点は軸上に来る（理想系）。数値的にも同じ位置になるはず。
"""

import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

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

def trace_system(wavelength_nm: float,
                 beam_diameter_mm: float,
                 doe_angle_deg: float,
                 lenses: List[Dict[str, float]]
                 ) -> Dict[str, object]:
    # DOE直後（z=0）で2本の光線を定義：中心(y=0)、角度 ±θ
    theta = math.radians(doe_angle_deg)
    ray_plus  = Ray(y=0.0, a=+theta)
    ray_minus = Ray(y=0.0, a=-theta)

    # 位置のタイムライン（DOE=0）
    z_curr = 0.0

    # 記録（可視化・デバッグ用）
    history_plus  = [{"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what":"DOE"}]
    history_minus = [{"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what":"DOE"}]

    # 各レンズへ順に伝搬→通過
    for i, lf in enumerate(lenses, start=1):
        zl = lf["z"]    # レンズ位置[mm]（DOE基準）
        f  = lf["f"]    # 焦点距離[mm]

        # 自由空間でその位置へ
        ray_plus  = propagate_to(z_curr, zl, ray_plus)
        ray_minus = propagate_to(z_curr, zl, ray_minus)
        z_curr = zl
        history_plus.append({"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what": f"before f{i}"})
        history_minus.append({"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what": f"before f{i}"})

        # レンズ通過
        ray_plus  = apply_M(M_lens(f), ray_plus)
        ray_minus = apply_M(M_lens(f), ray_minus)
        history_plus.append({"z": z_curr, "y": ray_plus.y,  "angle_deg": math.degrees(ray_plus.a),  "what": f"after f{i}"})
        history_minus.append({"z": z_curr, "y": ray_minus.y, "angle_deg": math.degrees(ray_minus.a), "what": f"after f{i}"})

    # ここで z_curr は f4 の位置（最後のレンズ位置）になっている前提
    # f4通過直後の角度（半角）と全角
    half_angle_after_f4_deg = abs(math.degrees(ray_plus.a))
    full_divergence_deg = 2.0 * half_angle_after_f4_deg

    # f4通過直後から、2本が交差する距離を計算
    # 対称系なら「軸(y=0)に交差」なので、任意片側で y + a*s = 0 → s = -y/a
    # （a=0 なら交差しない＝平行）
    if abs(ray_plus.a) < 1e-12:
        s_after_f4 = math.inf
    else:
        s_after_f4 = - ray_plus.y / ray_plus.a   # [mm]（f4直後からの距離）
    z_cross = z_curr + s_after_f4               # DOE基準の交差位置[mm]

    return {
        "inputs": {
            "wavelength_nm": wavelength_nm,
            "beam_diameter_mm": beam_diameter_mm,
            "doe_angle_deg": doe_angle_deg,
            "lenses": lenses
        },
        "results": {
            "half_angle_after_f4_deg": half_angle_after_f4_deg,
            "full_divergence_deg": full_divergence_deg,
            "distance_after_f4_to_cross_mm": s_after_f4,
            "z_cross_from_DOE_mm": z_cross
        },
        "history_plus": history_plus,
        "history_minus": history_minus
    }

# ===== 使い方例 =====
if __name__ == "__main__":
    # 図のように DOE→f1→f2→f3→f4 の順で位置[z]と焦点距離[f]を与える
    # （数値はダミー。お手元の設計値に置き換えてください）
    wavelength_nm = 515.0
    beam_diameter_mm = 2.2
    doe_angle_deg = 0.485

    lenses = [
        {"z": 300.0, "f": 300.0},  # f1 at z=300 mm, f=300 mm
        {"z": 660.0, "f": 60.0},   # f2 at z=600 mm, f=60 mm
        {"z": 920.0, "f": 200.0},  # f3 at z=800 mm, f=200 mm
        {"z": 1140.0,"f": 20.0},   # f4 at z=1020 mm, f=20 mm
    ]

    out = trace_system(wavelength_nm, beam_diameter_mm, doe_angle_deg, lenses)
    from pprint import pprint
    pprint(out["results"])

    # 必要なら履歴をCSV等に出力して可視化も可能（matplotlibで線図化など）
