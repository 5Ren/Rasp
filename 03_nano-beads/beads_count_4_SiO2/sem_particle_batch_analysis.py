#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEM particle batch analysis
===========================

Expected BMP geometry
---------------------
- 2560 x 2052 px
- Calibration: 200 px = 100 nm -> 0.5 nm/px
- Analysis image: X = 0..2559, Y = 0..1919
- Y >= 1920 is the SEM information band and is excluded.

Main processing
---------------
1. Crop the information band.
2. Gaussian smoothing: sigma = 2 px = 1 nm.
3. Estimate slowly-varying background with Gaussian sigma = 100 px = 50 nm.
4. Local contrast = smoothed - background.
5. Estimate local-contrast spread robustly with MAD.
6. Hysteresis threshold:
      low  = median + 1.2 * robust_sigma
      high = median + 3.0 * robust_sigma
   Pixels above the high threshold act as reliable seeds. Connected pixels
   above the low threshold are included in the same candidate.
7. 3x3 binary opening to remove isolated jagged pixels.
8. Candidate filtering:
      equivalent circular diameter: 10..200 nm
      circularity >= 0.35
      solidity >= 0.65
9. Objects touching the OUTER image border are visible in overlays but are
   excluded from quantitative particle-size statistics.
10. The image is divided into 2 x 2 equal subregions. A particle is assigned
    to Q1..Q4 by its centroid. This n=4 represents within-image spatial
    variability, not four independent SEM fields.

Outputs
-------
<output folder>/
  analysis_settings.txt
  summary.xlsx
  image_summary.csv
  quadrant_summary.csv
  all_particles.csv
  <image_stem>/
    00_original_crop.png
    01_smoothed.png
    02_background.png
    03_local_contrast.png
    04_binary_hysteresis_raw.png
    05_binary_cleaned.png
    06_particle_overlay.png
    07_quadrant_overlay.png
    08_size_histogram.png
    particles.csv
    quadrant_summary.csv
    image_summary.csv

Run
---
  py sem_particle_batch_analysis.py "C:\\path\\to\\bmp_folder"

If no input folder is given, a Windows folder-selection dialog is opened.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops


# -----------------------------------------------------------------------------
# Fixed image calibration / processing parameters
# -----------------------------------------------------------------------------
PIXEL_NM = 0.5
EXPECTED_WIDTH_PX = 2560
EXPECTED_HEIGHT_PX = 2052
ANALYSIS_HEIGHT_PX = 1920

SMOOTH_SIGMA_PX = 2.0
BACKGROUND_SIGMA_PX = 100.0
LOW_K = 1.2
HIGH_K = 3.0

MIN_EQ_DIAMETER_NM = 10.0
MAX_EQ_DIAMETER_NM = 200.0
MIN_CIRCULARITY = 0.35
MIN_SOLIDITY = 0.65

# Overlay colors are deliberately fixed only for semantic annotation.
COLOR_ACCEPTED = (0, 0, 255)       # OpenCV BGR: red
COLOR_EDGE = (0, 255, 255)         # yellow
COLOR_QUADRANT = (255, 255, 0)     # cyan


@dataclass
class ParticleRecord:
    source_file: str
    label: int
    subregion: int
    measurement_valid: bool
    touches_outer_border: bool
    area_px: float
    area_nm2: float
    perimeter_px: float
    eq_diameter_px: float
    eq_diameter_nm: float
    circularity: float
    solidity: float
    eccentricity: float
    centroid_x_px: float
    centroid_y_px: float
    centroid_x_nm: float
    centroid_y_nm: float
    mean_intensity: float
    max_intensity: float


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def parse_filename(stem: str) -> dict[str, Any]:
    """Best-effort parsing of names such as 600kHz_1000shots_8J_x100k_s2."""
    out: dict[str, Any] = {
        "frequency_text": "",
        "frequency_khz": None,
        "shots": None,
        "fluence_j_cm2": None,
    }

    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*(kHz|MHz)", stem)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        out["frequency_text"] = m.group(0)
        out["frequency_khz"] = val * (1000.0 if unit == "mhz" else 1.0)

    m = re.search(r"(?i)(\d+)\s*shots", stem)
    if m:
        out["shots"] = int(m.group(1))

    m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*J(?:cm2|/cm2)?", stem)
    if m:
        out["fluence_j_cm2"] = float(m.group(1))

    return out


def robust_sigma_mad(arr: np.ndarray) -> tuple[float, float]:
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, 1.4826 * mad


def normalize_u8(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def save_gray(path: Path, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8:
        arr = normalize_u8(arr)
    Image.fromarray(arr).save(path)


def save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def ensure_output_dir(input_dir: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        output_dir = explicit_output
    else:
        base = input_dir / "particle_analysis_output"
        if not base.exists():
            output_dir = base
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = input_dir / f"particle_analysis_output_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def find_bmps(input_dir: Path, recursive: bool) -> list[Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [p for p in candidates if p.is_file() and p.suffix.lower() == ".bmp"]
    return sorted(files, key=lambda p: p.name.lower())


def pick_input_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="Select folder containing SEM BMP images")
    root.destroy()
    return Path(selected) if selected else None


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return mean, sd


def percentile_or_nan(values: Iterable[float], q: float) -> float:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Minimal XLSX writer using only the Python standard library.
# This avoids requiring openpyxl/xlsxwriter on the user's PC.
# -----------------------------------------------------------------------------
def _excel_col_name(n1: int) -> str:
    out = ""
    n = n1
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _cell_xml(row: int, col: int, value: Any, style: int = 0) -> str:
    ref = f"{_excel_col_name(col)}{row}"
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        fv = float(value)
        if not math.isfinite(fv):
            return f'<c r="{ref}"{style_attr}/>'
        return f'<c r="{ref}"{style_attr}><v>{fv}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def _sheet_xml(headers: list[str], rows: list[dict[str, Any]]) -> str:
    all_rows: list[list[Any]] = [headers]
    for r in rows:
        all_rows.append([r.get(h) for h in headers])

    xml_rows = []
    for ridx, vals in enumerate(all_rows, start=1):
        style = 1 if ridx == 1 else 0
        cells = "".join(_cell_xml(ridx, cidx, val, style) for cidx, val in enumerate(vals, start=1))
        xml_rows.append(f'<row r="{ridx}">{cells}</row>')

    max_col = max(1, len(headers))
    max_row = max(1, len(all_rows))
    dimension = f"A1:{_excel_col_name(max_col)}{max_row}"

    # Fixed, readable widths. Excel can still auto-fit manually if desired.
    cols = ''.join(
        f'<col min="{i}" max="{i}" width="{24 if i == 1 else 16}" customWidth="1"/>'
        for i in range(1, max_col + 1)
    )

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>{cols}</cols>
  <sheetData>{''.join(xml_rows)}</sheetData>
  <autoFilter ref="{dimension}"/>
</worksheet>'''


def write_simple_xlsx(path: Path, sheets: list[tuple[str, list[str], list[dict[str, Any]]]]) -> None:
    # Sanitize and de-duplicate sheet names.
    safe_names: list[str] = []
    used: set[str] = set()
    for name, _, _ in sheets:
        n = re.sub(r"[\\/*?:\[\]]", "_", name)[:31] or "Sheet"
        base = n
        suffix = 1
        while n.lower() in used:
            suffix += 1
            tail = f"_{suffix}"
            n = (base[:31 - len(tail)] + tail)
        used.add(n.lower())
        safe_names.append(n)

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for i in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append('</Types>')

    workbook_sheets = ''.join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(safe_names, start=1)
    )
    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{workbook_sheets}</sheets>
</workbook>'''

    wb_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for i in range(1, len(sheets) + 1):
        wb_rels.append(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        )
    wb_rels.append(
        f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    wb_rels.append('</Relationships>')

    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ''.join(content_types))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", ''.join(wb_rels))
        z.writestr("xl/styles.xml", styles_xml)
        for i, (_, headers, rows) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(headers, rows))


# -----------------------------------------------------------------------------
# Main image analysis
# -----------------------------------------------------------------------------
def analyze_image(path: Path, image_out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_out.mkdir(parents=True, exist_ok=False)

    img_full = np.asarray(Image.open(path).convert("L"))
    full_h, full_w = img_full.shape

    if full_w != EXPECTED_WIDTH_PX or full_h != EXPECTED_HEIGHT_PX:
        print(
            f"WARNING: {path.name}: expected {EXPECTED_WIDTH_PX}x{EXPECTED_HEIGHT_PX}, "
            f"got {full_w}x{full_h}. Cropping first {ANALYSIS_HEIGHT_PX} rows if possible.",
            file=sys.stderr,
        )
    if full_h < ANALYSIS_HEIGHT_PX:
        raise ValueError(f"{path.name}: image height {full_h} < {ANALYSIS_HEIGHT_PX}")

    img = img_full[:ANALYSIS_HEIGHT_PX, :].copy()
    h, w = img.shape
    save_gray(image_out / "00_original_crop.png", img)

    # 1) Light Gaussian smoothing (sigma=1 nm)
    smoothed = cv2.GaussianBlur(
        img, (0, 0), sigmaX=SMOOTH_SIGMA_PX, sigmaY=SMOOTH_SIGMA_PX
    )
    save_gray(image_out / "01_smoothed.png", smoothed)

    # 2) Slowly varying local background (sigma=50 nm)
    background = cv2.GaussianBlur(
        smoothed, (0, 0), sigmaX=BACKGROUND_SIGMA_PX, sigmaY=BACKGROUND_SIGMA_PX
    )
    save_gray(image_out / "02_background.png", background)

    # 3) Local contrast
    residual = smoothed.astype(np.float32) - background.astype(np.float32)
    save_gray(image_out / "03_local_contrast.png", normalize_u8(residual, 1, 99))

    # 4) Robust scale and hysteresis binarization
    residual_median, robust_sigma = robust_sigma_mad(residual)
    low_threshold = residual_median + LOW_K * robust_sigma
    high_threshold = residual_median + HIGH_K * robust_sigma

    raw_mask = apply_hysteresis_threshold(residual, low_threshold, high_threshold)
    save_binary(image_out / "04_binary_hysteresis_raw.png", raw_mask)

    # 5) Clean isolated/jagged pixels; size filtering is done on components below.
    clean_mask = ndi.binary_opening(raw_mask, structure=np.ones((3, 3), dtype=bool))
    save_binary(image_out / "05_binary_cleaned.png", clean_mask)

    labels = label(clean_mask)
    accepted: list[ParticleRecord] = []

    for r in regionprops(labels, intensity_image=img):
        area_px = float(r.area)
        perimeter_px = float(r.perimeter)
        eq_px = float(r.equivalent_diameter_area)
        eq_nm = eq_px * PIXEL_NM
        circularity = 4.0 * math.pi * area_px / max(perimeter_px, 1.0) ** 2
        solidity = float(r.solidity)

        if not (MIN_EQ_DIAMETER_NM <= eq_nm <= MAX_EQ_DIAMETER_NM):
            continue
        if circularity < MIN_CIRCULARITY:
            continue
        if solidity < MIN_SOLIDITY:
            continue

        minr, minc, maxr, maxc = r.bbox
        touches_outer_border = minr <= 0 or minc <= 0 or maxr >= h or maxc >= w
        cy, cx = r.centroid
        quadrant = 1 + int(cx >= w / 2) + 2 * int(cy >= h / 2)

        accepted.append(
            ParticleRecord(
                source_file=path.name,
                label=int(r.label),
                subregion=quadrant,
                measurement_valid=not touches_outer_border,
                touches_outer_border=touches_outer_border,
                area_px=area_px,
                area_nm2=area_px * PIXEL_NM**2,
                perimeter_px=perimeter_px,
                eq_diameter_px=eq_px,
                eq_diameter_nm=eq_nm,
                circularity=float(circularity),
                solidity=solidity,
                eccentricity=float(r.eccentricity),
                centroid_x_px=float(cx),
                centroid_y_px=float(cy),
                centroid_x_nm=float(cx * PIXEL_NM),
                centroid_y_nm=float(cy * PIXEL_NM),
                mean_intensity=float(r.intensity_mean),
                max_intensity=float(r.intensity_max),
            )
        )

    accepted_labels = {p.label for p in accepted}
    valid_labels = {p.label for p in accepted if p.measurement_valid}
    edge_labels = accepted_labels - valid_labels

    # 6) Particle overlay without quadrants
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    valid_mask = np.isin(labels, list(valid_labels)).astype(np.uint8)
    edge_mask = np.isin(labels, list(edge_labels)).astype(np.uint8)
    valid_contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, valid_contours, -1, COLOR_ACCEPTED, 2)
    cv2.drawContours(overlay, edge_contours, -1, COLOR_EDGE, 2)
    cv2.imwrite(str(image_out / "06_particle_overlay.png"), overlay)

    # 7) Same overlay with Q1-Q4 boundaries
    qoverlay = overlay.copy()
    cv2.line(qoverlay, (w // 2, 0), (w // 2, h - 1), COLOR_QUADRANT, 2)
    cv2.line(qoverlay, (0, h // 2), (w - 1, h // 2), COLOR_QUADRANT, 2)
    for text, xy in [
        ("Q1", (20, 45)),
        ("Q2", (w // 2 + 20, 45)),
        ("Q3", (20, h // 2 + 45)),
        ("Q4", (w // 2 + 20, h // 2 + 45)),
    ]:
        cv2.putText(qoverlay, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(qoverlay, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(image_out / "07_quadrant_overlay.png"), qoverlay)

    # Quantitative records only use particles that do not touch the outer image border.
    valid_particles = [p for p in accepted if p.measurement_valid]
    valid_dicts = [asdict(p) for p in valid_particles]
    candidate_dicts = [asdict(p) for p in accepted]

    parsed = parse_filename(path.stem)
    full_area_um2 = w * h * (PIXEL_NM / 1000.0) ** 2
    quadrant_area_um2 = full_area_um2 / 4.0

    # Per-quadrant statistics
    quadrant_rows: list[dict[str, Any]] = []
    quadrant_means = []
    quadrant_densities = []

    for q in range(1, 5):
        qparts = [p for p in valid_particles if p.subregion == q]
        diam = np.asarray([p.eq_diameter_nm for p in qparts], dtype=float)
        count = len(qparts)
        density = count / quadrant_area_um2
        qmean = float(np.mean(diam)) if count else float("nan")
        qsd = float(np.std(diam, ddof=1)) if count > 1 else float("nan")
        coverage = sum(p.area_nm2 for p in qparts) / (quadrant_area_um2 * 1e6) * 100.0
        quadrant_means.append(qmean)
        quadrant_densities.append(density)

        quadrant_rows.append({
            "source_file": path.name,
            **parsed,
            "subregion": q,
            "subregion_area_um2": quadrant_area_um2,
            "particle_count": count,
            "density_per_um2": density,
            "mean_eq_diameter_nm": qmean,
            "sd_eq_diameter_nm": qsd,
            "median_eq_diameter_nm": float(np.median(diam)) if count else float("nan"),
            "D10_nm": percentile_or_nan(diam, 10),
            "D50_nm": percentile_or_nan(diam, 50),
            "D90_nm": percentile_or_nan(diam, 90),
            "coverage_percent": coverage,
        })

    # Full-image distribution statistics
    diam_all = np.asarray([p.eq_diameter_nm for p in valid_particles], dtype=float)
    density_mean, density_sd = mean_sd(quadrant_densities)
    qmean_mean, qmean_sd = mean_sd(quadrant_means)

    image_row = {
        "source_file": path.name,
        **parsed,
        "source_width_px": full_w,
        "source_height_px": full_h,
        "analysis_width_px": w,
        "analysis_height_px": h,
        "pixel_size_nm": PIXEL_NM,
        "analysis_area_um2": full_area_um2,
        "quadrant_area_um2": quadrant_area_um2,
        "accepted_candidates_including_outer_edge": len(accepted),
        "valid_particle_count": len(valid_particles),
        "full_image_density_per_um2": len(valid_particles) / full_area_um2,
        "quadrant_density_mean_per_um2": density_mean,
        "quadrant_density_sd_per_um2": density_sd,
        "full_particle_mean_eq_diameter_nm": float(np.mean(diam_all)) if len(diam_all) else float("nan"),
        "full_particle_sd_eq_diameter_nm": float(np.std(diam_all, ddof=1)) if len(diam_all) > 1 else float("nan"),
        "full_particle_median_eq_diameter_nm": float(np.median(diam_all)) if len(diam_all) else float("nan"),
        "D10_nm": percentile_or_nan(diam_all, 10),
        "D50_nm": percentile_or_nan(diam_all, 50),
        "D90_nm": percentile_or_nan(diam_all, 90),
        "quadrant_mean_diameter_mean_nm": qmean_mean,
        "quadrant_mean_diameter_sd_nm": qmean_sd,
        "coverage_percent": sum(p.area_nm2 for p in valid_particles) / (full_area_um2 * 1e6) * 100.0,
        "residual_median_gray": residual_median,
        "robust_sigma_gray": robust_sigma,
        "low_threshold_gray": low_threshold,
        "high_threshold_gray": high_threshold,
    }

    # 8) Histogram. Use all valid particles across all four subregions.
    plt.figure(figsize=(7.5, 5.0))
    if len(diam_all):
        # Freedman-Diaconis-like automatic bin selection via numpy/matplotlib.
        plt.hist(diam_all, bins="auto")
        plt.axvline(float(np.median(diam_all)), linestyle="--", linewidth=1.2, label=f"Median = {np.median(diam_all):.1f} nm")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No valid particles", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlabel("Equivalent circular diameter (nm)")
    plt.ylabel("Particle count")
    plt.title(path.stem)
    plt.tight_layout()
    plt.savefig(image_out / "08_size_histogram.png", dpi=200)
    plt.close()

    write_csv(image_out / "particles.csv", candidate_dicts, list(asdict(ParticleRecord("",0,0,False,False,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).keys()))
    write_csv(image_out / "quadrant_summary.csv", quadrant_rows)
    write_csv(image_out / "image_summary.csv", [image_row])

    return valid_dicts, quadrant_rows, image_row


def settings_rows() -> list[dict[str, Any]]:
    return [
        {"parameter": "PIXEL_NM", "value": PIXEL_NM, "meaning": "200 px = 100 nm -> 0.5 nm/px"},
        {"parameter": "ANALYSIS_HEIGHT_PX", "value": ANALYSIS_HEIGHT_PX, "meaning": "Use Y=0..1919; Y>=1920 excluded"},
        {"parameter": "SMOOTH_SIGMA_PX", "value": SMOOTH_SIGMA_PX, "meaning": f"Gaussian smoothing; {SMOOTH_SIGMA_PX*PIXEL_NM:g} nm"},
        {"parameter": "BACKGROUND_SIGMA_PX", "value": BACKGROUND_SIGMA_PX, "meaning": f"Local-background Gaussian; {BACKGROUND_SIGMA_PX*PIXEL_NM:g} nm"},
        {"parameter": "LOW_K", "value": LOW_K, "meaning": "Hysteresis low threshold = median + LOW_K * robust sigma"},
        {"parameter": "HIGH_K", "value": HIGH_K, "meaning": "Hysteresis high threshold = median + HIGH_K * robust sigma"},
        {"parameter": "MIN_EQ_DIAMETER_NM", "value": MIN_EQ_DIAMETER_NM, "meaning": "Minimum accepted equivalent circular diameter"},
        {"parameter": "MAX_EQ_DIAMETER_NM", "value": MAX_EQ_DIAMETER_NM, "meaning": "Maximum accepted equivalent circular diameter"},
        {"parameter": "MIN_CIRCULARITY", "value": MIN_CIRCULARITY, "meaning": "Candidate shape filter"},
        {"parameter": "MIN_SOLIDITY", "value": MIN_SOLIDITY, "meaning": "Candidate shape filter"},
        {"parameter": "n_definition", "value": 4, "meaning": "2x2 subregions within one SEM image; spatial variability, not independent fields"},
    ]


def write_settings_txt(path: Path) -> None:
    lines = [
        "SEM particle analysis settings",
        "==============================",
        "",
        f"Pixel calibration: {PIXEL_NM} nm/pixel (200 px = 100 nm)",
        f"Analyzed rows: Y = 0..{ANALYSIS_HEIGHT_PX-1}; Y >= {ANALYSIS_HEIGHT_PX} excluded",
        f"Gaussian smoothing sigma: {SMOOTH_SIGMA_PX} px = {SMOOTH_SIGMA_PX*PIXEL_NM} nm",
        f"Local-background Gaussian sigma: {BACKGROUND_SIGMA_PX} px = {BACKGROUND_SIGMA_PX*PIXEL_NM} nm",
        f"Hysteresis low threshold: median + {LOW_K} * robust_sigma",
        f"Hysteresis high threshold: median + {HIGH_K} * robust_sigma",
        "robust_sigma = 1.4826 * median(|local_contrast - median(local_contrast)|)",
        f"Accepted equivalent diameter: {MIN_EQ_DIAMETER_NM}..{MAX_EQ_DIAMETER_NM} nm",
        f"Minimum circularity: {MIN_CIRCULARITY}",
        f"Minimum solidity: {MIN_SOLIDITY}",
        "Outer-image-border-touching particles: shown, but excluded from quantitative size statistics",
        "Quadrants: Q1 upper-left, Q2 upper-right, Q3 lower-left, Q4 lower-right",
        "Particle-to-quadrant assignment: centroid position",
        "n=4 interpretation: within-image spatial variability, NOT four independent SEM fields",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch SEM particle analysis for BMP images")
    parser.add_argument("input_folder", nargs="?", type=Path, help="Folder containing BMP images")
    parser.add_argument("--output", type=Path, default=None, help="Optional output folder path")
    parser.add_argument("--recursive", action="store_true", help="Also search BMP files in subfolders")
    args = parser.parse_args()

    input_dir = args.input_folder
    if input_dir is None:
        input_dir = pick_input_folder()
        if input_dir is None:
            print("No input folder selected. Provide a folder on the command line.", file=sys.stderr)
            return 2

    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr)
        return 2

    bmps = find_bmps(input_dir, args.recursive)
    if not bmps:
        print(f"No BMP files found in: {input_dir}", file=sys.stderr)
        return 2

    output_dir = ensure_output_dir(input_dir, args.output.expanduser().resolve() if args.output else None)
    write_settings_txt(output_dir / "analysis_settings.txt")

    all_particles: list[dict[str, Any]] = []
    all_quadrants: list[dict[str, Any]] = []
    all_images: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"BMP files    : {len(bmps)}")
    print()

    for index, bmp in enumerate(bmps, start=1):
        print(f"[{index:02d}/{len(bmps):02d}] {bmp.name}")
        image_out = output_dir / bmp.stem
        try:
            particles, quadrants, image_summary = analyze_image(bmp, image_out)
            all_particles.extend(particles)
            all_quadrants.extend(quadrants)
            all_images.append(image_summary)
            print(
                f"    valid particles={image_summary['valid_particle_count']}, "
                f"density={image_summary['full_image_density_per_um2']:.2f}/um^2, "
                f"median={image_summary['full_particle_median_eq_diameter_nm']:.2f} nm"
            )
        except Exception as exc:
            errors.append({"source_file": bmp.name, "error": repr(exc)})
            print(f"    ERROR: {exc}", file=sys.stderr)

    image_headers = list(all_images[0].keys()) if all_images else ["source_file"]
    quadrant_headers = list(all_quadrants[0].keys()) if all_quadrants else ["source_file"]
    particle_headers = list(all_particles[0].keys()) if all_particles else list(asdict(ParticleRecord("",0,0,False,False,0,0,0,0,0,0,0,0,0,0,0,0,0,0)).keys())
    settings = settings_rows()

    write_csv(output_dir / "image_summary.csv", all_images, image_headers)
    write_csv(output_dir / "quadrant_summary.csv", all_quadrants, quadrant_headers)
    write_csv(output_dir / "all_particles.csv", all_particles, particle_headers)
    if errors:
        write_csv(output_dir / "errors.csv", errors)

    # Excel workbook contains four sheets.
    write_simple_xlsx(
        output_dir / "summary.xlsx",
        [
            ("Image Summary", image_headers, all_images),
            ("Quadrant Summary", quadrant_headers, all_quadrants),
            ("Particle Data", particle_headers, all_particles),
            ("Settings", ["parameter", "value", "meaning"], settings),
        ],
    )

    print()
    print("Done.")
    print(f"Output: {output_dir}")
    if errors:
        print(f"Completed with {len(errors)} error(s). See errors.csv")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
