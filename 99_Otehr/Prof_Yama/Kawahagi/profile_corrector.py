import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===== User settings =====
input_path = Path("./prof_data/unp.csv")       # input CSV file
x_col = "x"                       # x column name
z_col = "z"                       # z column name
waviness_window_ratio = 0.10       # larger = stronger waviness removal
# =========================

base = input_path.with_suffix("")

with open(input_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

x = np.array([float(r[x_col]) for r in rows], dtype=float)
z = np.array([float(r[z_col]) for r in rows], dtype=float)

order = np.argsort(x)
x = x[order]
z = z[order]

# 1. Tilt correction: linear least-squares fitting
coef = np.polyfit(x, z, 1)
tilt = np.polyval(coef, x)
z_tilt_corrected = z - tilt
z_tilt_corrected -= np.mean(z_tilt_corrected)

# 2. Waviness removal: subtract moving-average component
n = len(z)
window = max(5, int(round(n * waviness_window_ratio)))
if window % 2 == 0:
    window += 1
pad = window // 2
kernel = np.ones(window) / window
z_pad = np.pad(z_tilt_corrected, pad_width=pad, mode="reflect")
waviness = np.convolve(z_pad, kernel, mode="valid")
z_corrected = z_tilt_corrected - waviness

# 3. Refactoring: set mean height to zero
z_corrected -= np.mean(z_corrected)

# Save corrected CSV
csv_out = base.with_name(base.name + "_corrected.csv")
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "z_original", "z_tilt_corrected", "waviness_component", "z_corrected"])
    writer.writerows(zip(x, z, z_tilt_corrected, waviness, z_corrected))

def save_plot(path, y, title, ylabel):
    plt.figure(figsize=(9, 4.8), dpi=180)
    plt.plot(x, y, linewidth=1.0)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

save_plot(base.with_name(base.name + "_Graph_original.png"), z, "Original profile", "z")
save_plot(base.with_name(base.name + "_Graph_tilt_corrected.png"), z_tilt_corrected, "Tilt-corrected profile (mean = 0)", "z")
save_plot(base.with_name(base.name + "_Graph_corrected.png"), z_corrected, f"Corrected roughness profile (tilt + waviness removed, window={window})", "z")

plt.figure(figsize=(9, 5.2), dpi=180)
plt.plot(x, z - np.mean(z), linewidth=0.9, label="Original, mean-centered")
plt.plot(x, z_tilt_corrected, linewidth=0.9, label="Tilt-corrected")
plt.plot(x, z_corrected, linewidth=0.9, label="Corrected roughness")
plt.xlabel("x")
plt.ylabel("z (mean-centered)")
plt.title("Profile correction comparison")
plt.grid(True, linewidth=0.4, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(base.with_name(base.name + "_Graph_comparison.png"))
plt.close()

print(f"Saved: {csv_out}")
print(f"Moving-average window: {window} points")
print(f"Tilt slope: {coef[0]}")
