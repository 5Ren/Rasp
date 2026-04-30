import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

# ===================== SETTINGS =====================

# Folder with SEM images
IMAGE_DIR = Path(r"C:\Users\YamaLab-38\Downloads\D75_x10\10mJ_x30k")
OUTPUT_DIR = IMAGE_DIR  # change if you want another folder

# Scale: 0.6 pixel = 1 nm  →  1 pixel = 1/0.6 nm
NM_PER_PIXEL = 1.0 / 0.6    # ≒ 1.6667 nm

# Size ranges [nm]
SMALL_MAX_DIAM_NM = 40.0     # small: < 40 nm
LARGE_MIN_DIAM_NM = 40.0     # large: ≥ 40 nm

# Merge tolerance [nm]
POS_TOL_NM = 10.0            # center distance within 10 nm
DIAM_TOL_NM = 10.0           # diameter difference within 10 nm

# Bottom area (scale bar, text) to ignore [pixels]
BOTTOM_IGNORE_PIXELS = 126

# Circularity thresholds
CIRC_SMALL = 0.80    # small particles: must be very round
CIRC_LARGE = 0.50    # large particles: allow more ellipse-like

# Valid extensions
VALID_EXTS = [".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]

# ====================================================


def nm_to_px(d_nm: float) -> float:
    """Convert nm → pixels."""
    return d_nm / NM_PER_PIXEL


def px_to_nm(d_px: float) -> float:
    """Convert pixels → nm."""
    return d_px * NM_PER_PIXEL


def detect_particles(img_gray: np.ndarray,
                     mode: str = "small") -> pd.DataFrame:
    """
    Detect bright particles on dark SEM background.
    mode = "small" or "large" でフィルタを切り替える。
    Returns DataFrame with geometry info.
    """
    h, w = img_gray.shape

    # --- 1. Preprocessing (different for small / large) ---

    if mode == "small":
        # 小さい粒子用：
        # 背景を強くぼかして引き算 → 小さなブロブだけ強調 (Top-hat 的)
        bg = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=8)
        high = cv2.subtract(img_gray, bg)

        # さらに軽くボカしてノイズ低減
        high_blur = cv2.GaussianBlur(high, (3, 3), 0)

        # Otsu で2値化（明るい粒子）
        _, binary = cv2.threshold(
            high_blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        circularity_min = CIRC_SMALL
        d_min_nm = 0.0
        d_max_nm = SMALL_MAX_DIAM_NM

    elif mode == "large":
        # 大きい粒子用：
        # 少しだけボカして、背景のムラを減らす程度
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        _, binary = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        circularity_min = CIRC_LARGE
        d_min_nm = LARGE_MIN_DIAM_NM
        d_max_nm = np.inf

    else:
        raise ValueError("mode must be 'small' or 'large'")

    # --- 2. Contour extraction ---
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    records = []

    # --- 3. Measure each contour ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue

        # Circularity = 4πA / P²
        circularity = 4.0 * math.pi * area / (perim * perim)
        if circularity < circularity_min:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Ignore bottom region (scale bar & text)
        if cy > (h - BOTTOM_IGNORE_PIXELS):
            continue

        # Equivalent circle radius / diameter (pixels)
        r_px = math.sqrt(area / math.pi)
        d_px = 2.0 * r_px

        # Convert to nm
        d_nm = px_to_nm(d_px)

        # Size filter by nm range
        if not (d_min_nm <= d_nm < d_max_nm):
            continue

        r_nm = px_to_nm(r_px)

        records.append({
            "cx_pixel": cx,
            "cy_pixel": cy,
            "radius_pixel": r_px,
            "diameter_pixel": d_px,
            "radius_nm": r_nm,
            "diameter_nm": d_nm,
            "area_pixel": area,
            "circularity": circularity,
        })

    return pd.DataFrame(records)


def merge_small_large(df_small: pd.DataFrame,
                      df_large: pd.DataFrame,
                      pos_tol_nm: float,
                      diam_tol_nm: float) -> pd.DataFrame:
    """
    small（<40nm）と large（>=40nm）の検出結果をマージする。
    - large は常に優先
    - small が large と (位置 & 直径) で近いものは「同一粒子」とみなして捨てる
    - どの large にもマッチしない small は small として残す
    """
    df_large = df_large.copy()
    df_large["scale_class"] = "large"

    keep_flags = np.ones(len(df_small), dtype=bool)

    for i, s in df_small.iterrows():
        sx, sy = s["cx_pixel"], s["cy_pixel"]
        sd = s["diameter_nm"]

        dx_nm = (df_large["cx_pixel"] - sx) * NM_PER_PIXEL
        dy_nm = (df_large["cy_pixel"] - sy) * NM_PER_PIXEL
        dist_nm = np.sqrt(dx_nm**2 + dy_nm**2)

        dd_nm = np.abs(df_large["diameter_nm"] - sd)

        # same particle?
        if ((dist_nm <= pos_tol_nm) & (dd_nm <= diam_tol_nm)).any():
            keep_flags[i] = False

    df_small_kept = df_small[keep_flags].copy()
    df_small_kept["scale_class"] = "small"

    df_merged = pd.concat([df_large, df_small_kept], ignore_index=True)

    # 連番 ID を振る
    df_merged = df_merged.sort_values(
        by=["scale_class", "diameter_nm"], ascending=[False, False]
    ).reset_index(drop=True)
    df_merged.insert(0, "id", df_merged.index + 1)

    return df_merged


def draw_overlay(img_color: np.ndarray,
                 df_merged: pd.DataFrame) -> np.ndarray:
    """
    元画像上に、検出した粒子を円＋idで描画する。
    large: green circle, small: cyan circle, text: red id
    """
    overlay = img_color.copy()

    for _, row in df_merged.iterrows():
        cx = row["cx_pixel"]
        cy = row["cy_pixel"]
        r_px = row["radius_pixel"]
        pid = int(row["id"])
        cls = row["scale_class"]

        center = (int(round(cx)), int(round(cy)))
        radius = int(round(r_px))

        # color by class
        if cls == "large":
            color = (0, 255, 0)    # green
        else:
            color = (255, 255, 0)  # cyan

        cv2.circle(overlay, center, radius, color, 1)

        # label
        label = str(pid)
        text_pos = (center[0] - 5, center[1] - radius - 3)
        cv2.putText(
            overlay, label, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return overlay


def process_one_image(img_path: Path):
    print(f"Processing: {img_path.name}")

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_gray is None or img_color is None:
        print(f"  [WARN] Cannot read image.")
        return

    # --- small / large detection ---
    df_small = detect_particles(img_gray, mode="small")
    df_large = detect_particles(img_gray, mode="large")

    # --- merge ---
    df_merged = merge_small_large(
        df_small, df_large,
        pos_tol_nm=POS_TOL_NM,
        diam_tol_nm=DIAM_TOL_NM
    )

    # --- save CSVs ---
    stem = img_path.stem

    out_small = OUTPUT_DIR / f"{stem}_small_particles.csv"
    out_large = OUTPUT_DIR / f"{stem}_large_particles.csv"
    out_merged = OUTPUT_DIR / f"{stem}_merged_particles.csv"

    df_small.to_csv(out_small, index=False)
    df_large.to_csv(out_large, index=False)
    df_merged.to_csv(out_merged, index=False)

    print(f"  small:  {len(df_small)} particles -> {out_small.name}")
    print(f"  large:  {len(df_large)} particles -> {out_large.name}")
    print(f"  merged: {len(df_merged)} particles -> {out_merged.name}")

    # --- overlay image ---
    overlay = draw_overlay(img_color, df_merged)
    out_overlay = OUTPUT_DIR / f"{stem}_merged_overlay.png"
    cv2.imwrite(str(out_overlay), overlay)
    print(f"  overlay saved: {out_overlay.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(IMAGE_DIR.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        process_one_image(img_path)


if __name__ == "__main__":
    main()
