
# -*- coding: utf-8 -*-
"""
Run NH4 Model 1 with robust data cleaning, alignment, plotting, and evaluation.
This script loads flow and NH4 data, cleans and merges them, runs the model,
and compares simulated vs measured concentrations with diagnostic plots and metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from NH4models import NH4inletModel1
import objective_functions as objFun

# ---------------------------------------------------------------------------
# 1. Settings: file paths and delimiter
# ---------------------------------------------------------------------------
FLOW_FILE = "flowdata_DAM_180715_180831.csv"
NH4_FILE = "NH4data_180604_180715.csv"
SEP = ";"  # CSV delimiter used in the raw files

# ---------------------------------------------------------------------------
# 2. Load raw data
# ---------------------------------------------------------------------------
flowData = pd.read_csv(FLOW_FILE, sep=SEP)
nh4Data = pd.read_csv(NH4_FILE, sep=SEP)

# Normalize column names to lowercase for consistency
flowData.columns = [c.strip().lower() for c in flowData.columns]
nh4Data.columns = [c.strip().lower() for c in nh4Data.columns]

# Ensure both datasets have a 'time' column
if 'time' not in flowData.columns or 'time' not in nh4Data.columns:
    raise ValueError("Both CSVs must have a 'time' column.")

# Rename columns to standard names if needed
if 'flow' not in flowData.columns:
    for cand in ['q', 'flow_m3h', 'flowrate']:
        if cand in flowData.columns:
            flowData.rename(columns={cand: 'flow'}, inplace=True)
            break
if 'nh4' not in nh4Data.columns:
    for cand in ['nh4_mg/l', 'nh4_conc', 'ammonium']:
        if cand in nh4Data.columns:
            nh4Data.rename(columns={cand: 'nh4'}, inplace=True)
            break

# Parse time columns into datetime objects
flowData['time'] = pd.to_datetime(flowData['time'])
nh4Data['time'] = pd.to_datetime(nh4Data['time'])

# Keep only the relevant columns
flowData = flowData[['time', 'flow']].copy()
nh4Data = nh4Data[['time', 'nh4']].copy()

# ---------------------------------------------------------------------------
# 3. Clean flow data
# ---------------------------------------------------------------------------
flowData = flowData.sort_values('time')
flowData['flow'] = flowData['flow'].astype(float).interpolate().bfill()
flowData['flow'] = flowData['flow'].rolling(window=15, center=True).median().bfill().ffill()
flowData['flow'] = flowData['flow'].clip(lower=0)

# ---------------------------------------------------------------------------
# 4. Clean NH4 data
# ---------------------------------------------------------------------------
nh4Data = nh4Data.sort_values('time')
nh4Data['nh4'] = pd.to_numeric(nh4Data['nh4'], errors='coerce')
nh4Data['nh4'] = nh4Data['nh4'].rolling(window=5, center=True).median().bfill().ffill()

# ---------------------------------------------------------------------------
# 5. Merge datasets by time
# ---------------------------------------------------------------------------
inputData = pd.merge(flowData, nh4Data, on='time', how='inner').reset_index(drop=True)

# Apply a longer median filter (window=60 points ~ 2 hours) to flow for final smoothing
inputData['flow'] = (
    inputData['flow']
    .rolling(window=60, center=True).median()
    .bfill().ffill()
)
# Clip flow to minimum of 1 m³/hr to avoid division by zero in the model
inputData['flow'] = inputData['flow'].clip(lower=1.0)

print(f"Merged rows: {inputData.shape[0]}")
if inputData.shape[0] == 0:
    raise ValueError("No overlapping timestamps between flow and NH4. Check file periods.")

# ---------------------------------------------------------------------------
# 6. Define model parameters
# ---------------------------------------------------------------------------
param = np.array([20000, 10000, -5000, 8000, -4000, 11500])
#param = np.array([113596,-44783,-19996,-11036,15235,11500])
# ---------------------------------------------------------------------------
# 7. Run the NH4 inlet model
# ---------------------------------------------------------------------------
out = NH4inletModel1(param, inputData)

# ---------------------------------------------------------------------------
# 8. Add baseline offset (+40 mg/L) to simulated concentrations
# ---------------------------------------------------------------------------
# This is equivalent to introducing a bias term k_bias = 40 mg/L
out['simNH4conc_shifted'] = out['simNH4conc'] 

# ---------------------------------------------------------------------------
# 9. Align observed and simulated data by time
# ---------------------------------------------------------------------------
comp = pd.merge(
    inputData[['time', 'nh4']],
    out[['time', 'simNH4conc_shifted']],
    on='time',
    how='inner'
).reset_index(drop=True)

print(f"Comparison rows: {comp.shape[0]}")
if comp.shape[0] == 0:
    raise ValueError("No overlapping timestamps after model run. Check time alignment.")

# ---------------------------------------------------------------------------
# 10. Plot results
# ---------------------------------------------------------------------------
# Full scale plot
plt.figure(figsize=(12, 5))
plt.plot(comp['time'], comp['nh4'], 'r.', markersize=2, label='Measured NH4')
plt.plot(comp['time'], comp['simNH4conc_shifted'], 'b-', linewidth=1.0, label='Simulated NH4 (Model 1) +40 mg/L')
plt.xlabel("Time")
plt.ylabel("NH4 concentration [mg/L]")
plt.title("Measured vs Simulated NH4 (full scale, with +40 mg/L offset)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Zoomed plot
plt.figure(figsize=(12, 5))
plt.plot(comp['time'], comp['nh4'], 'r.', markersize=2, label='Measured NH4')
plt.plot(comp['time'], comp['simNH4conc_shifted'], 'b-', linewidth=1.0, label='Simulated NH4 (Model 1) +40 mg/L')
plt.xlabel("Time")
plt.ylabel("NH4 concentration [mg/L]")
plt.title("Measured vs Simulated NH4 (zoomed, with +40 mg/L offset)")
plt.ylim(0, 100)  # adjust zoom range as needed
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 11. Evaluate performance with objective functions
# ---------------------------------------------------------------------------
mare = objFun.MARE(comp['nh4'].values, comp['simNH4conc_shifted'].values)
rmse = objFun.RMSE(comp['nh4'].values, comp['simNH4conc_shifted'].values)
inv_mse = objFun.invMSE(comp['nh4'].values, comp['simNH4conc_shifted'].values)

print(f"MARE: {mare}")
print(f"RMSE: {rmse}")
print(f"invMSE: {inv_mse}")

# ---------------------------------------------------------------------------
# 12. Diagnostics: ranges of data and simulation
# ---------------------------------------------------------------------------
print("Flow range:", inputData['flow'].min(), inputData['flow'].max())
print("NH4 observed range:", inputData['nh4'].min(), inputData['nh4'].max())
print("Simulated load range:", out['simNH4load'].min(), out['simNH4load'].max())
print("Simulated conc range (shifted):", out['simNH4conc_shifted'].min(), out['simNH4conc_shifted'].max())