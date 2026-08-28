#!/usr/bin/env python3
"""
SEM particle analysis prototype.

Calibration
-----------
200 px = 100 nm -> 0.5 nm/px

Analysis area
-------------
BMP size: 2560 x 2052 px
Use Y = 0..1919 only.
Y >= 1920 is the information band and is excluded.

Processing
----------
1. Gaussian smoothing: sigma = 2 px = 1 nm
2. Local background: Gaussian sigma = 100 px = 50 nm
3. Local contrast = smoothed - background
4. robust_sigma = 1.4826 * MAD(local contrast)
5. Hysteresis threshold:
       low  = median + 1.2 * robust_sigma
       high = median + 3.0 * robust_sigma
   Pixels above high are reliable seeds.
   Connected pixels above low are allowed to join those seeds.
6. Remove components smaller than a 10-nm equivalent-circle area.
7. Candidate filtering:
       equivalent diameter = 10..200 nm
       circularity >= 0.35
       solidity >= 0.65
8. Objects touching the OUTER image border are shown but excluded from
   preliminary size statistics.

The 2x2 subregions are assigned from particle centroid position.
"""

from pathlib import Path
import argparse
import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects


PIXEL_NM = 0.5
ANALYSIS_HEIGHT_PX = 1920

SMOOTH_SIGMA_PX = 2.0
BACKGROUND_SIGMA_PX = 100.0

LOW_K = 1.2
HIGH_K = 3.0

MIN_EQ_DIAMETER_NM = 10.0
MAX_EQ_DIAMETER_NM = 200.0
MIN_CIRCULARITY = 0.35
MIN_SOLIDITY = 0.65


def analyze(input_path: Path, output_dir: Path) -> None:
    img_full = np.asarray(Image.open(input_path).convert("L"))
    if img_full.shape[0] < ANALYSIS_HEIGHT_PX:
        raise ValueError(
            f"Image is only {img_full.shape[0]} px high; "
            f"{ANALYSIS_HEIGHT_PX} px are required."
        )

    img = img_full[:ANALYSIS_HEIGHT_PX, :].copy()
    h, w = img.shape

    # 1) light noise suppression
    smoothed = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=SMOOTH_SIGMA_PX,
        sigmaY=SMOOTH_SIGMA_PX,
    )

    # 2) slowly varying local background
    background = cv2.GaussianBlur(
        smoothed,
        (0, 0),
        sigmaX=BACKGROUND_SIGMA_PX,
        sigmaY=BACKGROUND_SIGMA_PX,
    )

    # 3) local contrast
    residual = (
        smoothed.astype(np.float32)
        - background.astype(np.float32)
    )

    # 4) robust noise scale
    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))
    robust_sigma = 1.4826 * mad

    low_threshold = residual_median + LOW_K * robust_sigma
    high_threshold = residual_median + HIGH_K * robust_sigma

    # 5) hysteresis threshold
    mask = apply_hysteresis_threshold(
        residual,
        low_threshold,
        high_threshold,
    )

    mask = ndi.binary_opening(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )

    min_area_px = math.pi * (
        MIN_EQ_DIAMETER_NM / 2.0 / PIXEL_NM
    ) ** 2

    mask = remove_small_objects(
        mask,
        min_size=max(1, int(round(min_area_px))),
    )

    # 6) connected components
    labels = label(mask)

    props = regionprops_table(
        labels,
        intensity_image=img,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "equivalent_diameter_area",
            "centroid",
            "bbox",
            "mean_intensity",
            "max_intensity",
        ),
    )

    df = pd.DataFrame(props)

    if len(df) == 0:
        raise RuntimeError("No connected components were detected.")

    df["circularity"] = (
        4.0 * np.pi * df["area"]
        / np.maximum(df["perimeter"], 1.0) ** 2
    )
    df["eq_diameter_nm"] = (
        df["equivalent_diameter_area"] * PIXEL_NM
    )
    df["area_nm2"] = df["area"] * PIXEL_NM ** 2

    df["centroid_x_px"] = df["centroid-1"]
    df["centroid_y_px"] = df["centroid-0"]
    df["centroid_x_nm"] = df["centroid_x_px"] * PIXEL_NM
    df["centroid_y_nm"] = df["centroid_y_px"] * PIXEL_NM

    df["touches_outer_border"] = (
        (df["bbox-0"] <= 0)
        | (df["bbox-1"] <= 0)
        | (df["bbox-2"] >= h)
        | (df["bbox-3"] >= w)
    )

    accepted = df[
        df["eq_diameter_nm"].between(
            MIN_EQ_DIAMETER_NM,
            MAX_EQ_DIAMETER_NM,
            inclusive="both",
        )
        & (df["circularity"] >= MIN_CIRCULARITY)
        & (df["solidity"] >= MIN_SOLIDITY)
    ].copy()

    accepted["measurement_valid"] = (
        ~accepted["touches_outer_border"]
    )

    # Q1 upper-left, Q2 upper-right,
    # Q3 lower-left, Q4 lower-right
    accepted["subregion"] = (
        1
        + (accepted["centroid_x_px"] >= w / 2).astype(int)
        + 2 * (accepted["centroid_y_px"] >= h / 2).astype(int)
    )

    # 7) overlay
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    valid_labels = accepted.loc[
        accepted["measurement_valid"], "label"
    ].astype(int)
    edge_labels = accepted.loc[
        ~accepted["measurement_valid"], "label"
    ].astype(int)

    valid_mask = np.isin(labels, valid_labels).astype(np.uint8)
    edge_mask = np.isin(labels, edge_labels).astype(np.uint8)

    valid_contours, _ = cv2.findContours(
        valid_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    edge_contours, _ = cv2.findContours(
        edge_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Red = accepted candidate
    cv2.drawContours(rgb, valid_contours, -1, (255, 0, 0), 2)

    # Yellow = touches the OUTER image edge
    cv2.drawContours(rgb, edge_contours, -1, (255, 255, 0), 2)

    # Cyan = 2x2 subdivision
    cv2.line(
        rgb,
        (w // 2, 0),
        (w // 2, h - 1),
        (0, 255, 255),
        2,
    )
    cv2.line(
        rgb,
        (0, h // 2),
        (w - 1, h // 2),
        (0, 255, 255),
        2,
    )

    # 8) preliminary statistics
    valid = accepted[accepted["measurement_valid"]].copy()

    area_um2 = (
        w * h * (PIXEL_NM / 1000.0) ** 2
    )

    summary = pd.DataFrame([{
        "input_file": input_path.name,
        "analysis_width_px": w,
        "analysis_height_px": h,
        "pixel_size_nm": PIXEL_NM,
        "analysis_area_um2": area_um2,
        "residual_median_gray": residual_median,
        "robust_sigma_gray": robust_sigma,
        "low_threshold_gray": low_threshold,
        "high_threshold_gray": high_threshold,
        "candidate_count": len(accepted),
        "measurement_count": len(valid),
        "mean_eq_diameter_nm":
            valid["eq_diameter_nm"].mean(),
        "median_eq_diameter_nm":
            valid["eq_diameter_nm"].median(),
        "sd_eq_diameter_nm":
            valid["eq_diameter_nm"].std(ddof=1),
        "density_per_um2":
            len(valid) / area_um2,
    }])

    # 9) save
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    Image.fromarray(rgb).save(
        output_dir / f"{stem}_particle_overlay.png"
    )
    accepted.to_csv(
        output_dir / f"{stem}_particle_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / f"{stem}_particle_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bmp", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
    )
    args = parser.parse_args()

    analyze(args.bmp, args.output_dir)


if __name__ == "__main__":
    main()
