import os
import sys
import numpy as np
import re

try:
    import pandas as pd
except ImportError:
    print("pandas is required to run this script. Install via 'pip install pandas' or add to requirements.txt.")
    raise

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- FILE PATHS ----------------

pred_file = "heart_rate_data.csv"
window_file = "hr_30sec_avg.csv"  # written by the GUI application

gt_file = r"C:\Users\tvina\Documents\1- Major Project\dataset\Unzipped-3\8-gt\gtdump.xmp"

# ---------------- LOAD PREDICTIONS ----------------

pred_windows = None  

if os.path.exists(window_file):
    df = pd.read_csv(window_file, encoding='utf-8')
    temp_windows = df["WindowAverageBPM"].values
    # only use if non-empty
    if len(temp_windows) > 0:
        pred_windows = temp_windows
        print(f"Loaded {len(pred_windows)} windowed predictions from {window_file}")

if pred_windows is None:
    # fall back to raw predictions
    if os.path.exists(window_file):
        print(f"Window file {window_file} is empty. Falling back to raw predictions...")
    df = pd.read_csv(pred_file, encoding='utf-8')
    pred_hr = df["Heart Rate (BPM)"].values
    print(f"Loaded {len(pred_hr)} raw predictions")

# ---------------- 30 SECOND AVERAGES (if not precomputed) ----------------

if pred_windows is None:
    window = 30
    pred_windows = np.array([
        np.mean(pred_hr[i:i+window])
        for i in range(0, len(pred_hr), window)
        if len(pred_hr[i:i+window]) == window
    ])
    print(f"Computed {len(pred_windows)} 30-second window averages")

# ---------------- LOAD GROUND TRUTH ----------------

with open(gt_file, "r") as f:
    lines = f.readlines()

# Parse CSV format: each line has comma-separated values
# Column 2 (index 1) = GT HR (the actual heart rate!)
gt_hr_list = []
for line in lines:
    line = line.strip()
    if line:  # skip empty lines
        try:
            vals = line.split(',')
            if len(vals) >= 2:
                hr = float(vals[1])  # column 2 (index 1) = GT HR
                gt_hr_list.append(hr)
        except ValueError:
            continue  # skip lines that can't be parsed

gt_hr = np.array(gt_hr_list)
print(f"Loaded {len(gt_hr)} GT HR values from column 2 (index 1)")
print(f"First 10 GT HR values: {gt_hr[:10]}")
print(f"Min GT HR: {np.min(gt_hr)}, Max GT HR: {np.max(gt_hr)}, Mean GT HR: {np.mean(gt_hr):.1f}")
if len(gt_hr) == 0:
    print("ERROR: No ground truth HR values found. Check the GT file format.")
    sys.exit(1)

# ---------------- RESAMPLE ----------------

# Determine the number of GT samples per prediction window
samples_per_window = len(gt_hr) // len(pred_windows)
print(f"Loaded {len(gt_hr)} GT HR values")
print(f"GT samples per prediction window: {samples_per_window}")

# Take the LAST GT sample from each window (the 30th second value)
gt_windows = []
for i in range(len(pred_windows)):
    # Index at end of each window
    idx = min((i + 1) * samples_per_window - 1, len(gt_hr) - 1)
    gt_windows.append(gt_hr[idx])

gt_windows = np.array(gt_windows)

print(f"\nPredicted HR per window: {pred_windows}")
print(f"GT HR (30th sec) per window: {gt_windows}")
print(f"Using GT indices: {[(i + 1) * samples_per_window - 1 for i in range(len(pred_windows))]}")

# Skip first window (warmup period with buffer artifacts)
# Use only windows 2 onwards for evaluation
if len(pred_windows) > 1:
    pred_windows_eval = pred_windows[1:]  # skip index 0
    gt_windows_eval = gt_windows[1:]      # skip index 0
    print(f"\nSkipping first window (warmup). Using windows 2-{len(pred_windows)}")
else:
    pred_windows_eval = pred_windows
    gt_windows_eval = gt_windows
    print("\nWarning: Only 1 window available, using it despite potential warmup artifacts")

print(f"Predicted HR (windows to evaluate): {pred_windows_eval}")
print(f"GT HR (windows to evaluate): {gt_windows_eval}")

# ---------------- METRICS ----------------

mae = mean_absolute_error(gt_windows_eval, pred_windows_eval)
rmse = np.sqrt(mean_squared_error(gt_windows_eval, pred_windows_eval))
if len(pred_windows_eval) > 1:
    corr = np.corrcoef(gt_windows_eval, pred_windows_eval)[0, 1]
else:
    corr = np.nan

# ---------------- OUTPUT ----------------

print("\n========== FINAL EVALUATION ==========")
print("Average Predicted HR:", round(np.mean(pred_windows_eval),2))
print("Average Ground Truth HR:", round(np.mean(gt_windows_eval),2))
print("MAE:", round(mae,2), "BPM")
print("RMSE:", round(rmse,2), "BPM")
print("======================================")