import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math

# ======== SETTINGS ========

# Folder with images
image_dir = Path(r"C:\Users\YamaLab-38\Downloads\D75_x10\10mJ_x30k")
output_dir = image_dir  # change if you want

# Circularity threshold (1.0 = perfect circle)
CIRCULARITY_THRESHOLD = 0.5

# Scale: 0.6 pixel = 1 nm  →  1 pixel = 1/0.6 nm
NM_PER_PIXEL = 1.0 / 0.6

# Minimum diameter to keep [nm]
MIN_DIAMETER_NM = 20.0

# Bottom area (scale bar, text) to ignore [pixels]
BOTTOM_IGNORE_PIXELS = 126

# Valid image extensions
VALID_EXTS = [".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]

# ===========================


def detect_round_particles_and_overlay(img_path: Path):
    """Detect bright round particles and return (DataFrame, overlay image)."""
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img_gray is None or img_color is None:
        print(f"[WARN] Cannot read image: {img_path}")
        return pd.DataFrame(), None

    h, w = img_gray.shape

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Otsu threshold: bright objects on dark background
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours (external)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    records = []
    overlay = img_color.copy()
    pid = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue

        # Circularity
        circularity = 4.0 * math.pi * area / (perim * perim)
        if circularity < CIRCULARITY_THRESHOLD:
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

        # Equivalent-circle radius / diameter (pixels)
        r_pix = math.sqrt(area / math.pi)
        d_pix = 2.0 * r_pix

        # Convert to nm
        r_nm = r_pix * NM_PER_PIXEL
        d_nm = d_pix * NM_PER_PIXEL

        # Diameter filter (nm)
        if d_nm < MIN_DIAMETER_NM:
            continue

        # ID for CSV & overlay
        pid += 1
        center_int = (int(round(cx)), int(round(cy)))
        radius_int = int(round(r_pix))

        # Draw circle (green)
        cv2.circle(overlay, center_int, radius_int, (0, 255, 0), 1)

        # Draw ID (red)
        label = str(pid)
        text_pos = (center_int[0] - 5, center_int[1] - radius_int - 3)
        cv2.putText(
            overlay, label, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        records.append({
            "id": pid,
            "cx_pixel": cx,
            "cy_pixel": cy,
            "radius_pixel": r_pix,
            "diameter_pixel": d_pix,
            "radius_nm": r_nm,
            "diameter_nm": d_nm,
            "area_pixel": area,
            "circularity": circularity,
        })

    df = pd.DataFrame(records)
    return df, overlay


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        print(f"Processing: {img_path.name}")

        df, overlay = detect_round_particles_and_overlay(img_path)

        # CSV
        out_csv = output_dir / f"{img_path.stem}_circles.csv"
        df.to_csv(out_csv, index=False)
        print(f"  -> {len(df)} particles saved to {out_csv.name}")

        # Overlay image
        if overlay is not None:
            out_img = output_dir / f"{img_path.stem}_circles_overlay.png"
            cv2.imwrite(str(out_img), overlay)
            print(f"  -> overlay image saved to {out_img.name}")


if __name__ == "__main__":
    main()
