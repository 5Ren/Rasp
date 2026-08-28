SEM particle batch analysis - Windows quick start
=================================================

1) Put these two files somewhere convenient:
   - sem_particle_batch_analysis.py
   - requirements_sem_particle_analysis.txt

2) Open PowerShell in that folder and check pip through Python:

   py -m pip --version

   If pip is missing:

   py -m ensurepip --upgrade
   py -m pip install --upgrade pip

3) Install the required packages:

   py -m pip install -r .\requirements_sem_particle_analysis.txt

   If the command 'py' itself is unavailable, replace 'py' with 'python'.

4) Run the analysis.

   Option A: folder picker
   -----------------------
   py .\sem_particle_batch_analysis.py

   A folder-selection dialog opens. Select the folder that contains the BMP files.

   Option B: specify the folder directly
   -------------------------------------
   py .\sem_particle_batch_analysis.py "C:\path\to\BMP_folder"

5) Output

   The script creates a new folder inside the selected input folder:

   particle_analysis_output\

   If that folder already exists, a timestamp is appended so old results are not overwritten.

   Top-level files:
   - analysis_settings.txt   Processing conditions in human-readable form
   - summary.xlsx            Image summary / Q1-Q4 summary / particle data / settings
   - image_summary.csv       One row per SEM image
   - quadrant_summary.csv    Four rows per SEM image (Q1-Q4)
   - all_particles.csv       One row per measured particle

   For every BMP, a subfolder is created containing:
   - 00_original_crop.png              Cropped SEM image, information band removed
   - 01_smoothed.png                   Gaussian smoothing result
   - 02_background.png                 Estimated slowly-varying background
   - 03_local_contrast.png             Smoothed image minus estimated background
   - 04_binary_hysteresis_raw.png      Raw hysteresis-threshold mask
   - 05_binary_cleaned.png             Mask after 3x3 binary opening
   - 06_particle_overlay.png           Detected particle contours over SEM image
   - 07_quadrant_overlay.png           Same contours plus Q1-Q4 division
   - 08_size_histogram.png             Equivalent circular diameter histogram
   - particles.csv                     Particle-level measurements for that image
   - quadrant_summary.csv              Q1-Q4 statistics for that image
   - image_summary.csv                 Whole-image statistics for that image

Fixed analysis assumptions
--------------------------
- Calibration: 200 px = 100 nm = 0.5 nm/px
- Source image expected: 2560 x 2052 px
- Analysis region: Y=0..1919
- Information band: Y>=1920 excluded
- 2 x 2 division: Q1 upper-left, Q2 upper-right, Q3 lower-left, Q4 lower-right
- n=4 is within-image spatial variability, not four independent SEM fields.

Current particle-detection parameters
-------------------------------------
- Gaussian smoothing sigma: 2 px = 1 nm
- Background Gaussian sigma: 100 px = 50 nm
- Hysteresis threshold:
    low  = median + 1.2 * robust_sigma
    high = median + 3.0 * robust_sigma
- Equivalent circular diameter: 10..200 nm
- Circularity >= 0.35
- Solidity >= 0.65
- Objects touching the OUTER image border are shown but excluded from quantitative size statistics.
