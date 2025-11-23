# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:26:42 2025

@author: rraat
"""
import os
os.chdir(r"C:\Users\rraat\OneDrive - Danmarks Tekniske Universitet\Documents\Academic\DTU\1ST SEMESTER\12104 Environmental Modelling\Module 4\Assignment 4")

# ---- Imports ----
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from NH4models import NH4inletModel0, NH4inletModel1, NH4inletModel3, NH4inletModel4

# ---- Load data ----
flowdata = pd.read_csv(
    'flowdata_DAM_180604_180715.csv',
    sep=';', parse_dates=['time']
)
NH4_data = pd.read_csv(
    'NH4data_180604_180715.csv',
    sep=';', parse_dates=['time']
)

# ---- Task 1.1: Calculate NH4 flux ----
NH4_flux = flowdata['flow'] * NH4_data['nh4']   # g/hr

# ---- Plot Flow ----
plt.figure(figsize=(10, 5))
plt.plot(flowdata['time'], flowdata['flow'],
         label='Flow (m3/s)', color='red', linestyle='-')
plt.xlabel('Time'); plt.ylabel('Flow (m3/s)')
plt.xticks(rotation=45); plt.legend(); plt.show()

# ---- Plot NH4 concentration ----
plt.figure(figsize=(10, 5))
plt.plot(NH4_data['time'], NH4_data['nh4'],
         label='NH4 (mg/l)', color='blue', linestyle='-')
plt.xlabel('Time'); plt.ylabel('NH4 (mg/l)')
plt.xticks(rotation=45); plt.legend(); plt.show()

# ---- Plot NH4 flux ----
plt.figure(figsize=(10, 5))
plt.plot(NH4_data['time'], NH4_flux,
         label='NH4 Flux (g/s)', color='green', linestyle='-')
plt.xlabel('Time'); plt.ylabel('NH4 Flux (g/s)')
plt.xticks(rotation=45); plt.legend(); plt.show()

# ---- Combined plot ----
plt.figure(figsize=(12, 6))
plt.plot(flowdata['time'], flowdata['flow'],
         label='Flow (m3/s)', color='blue', linestyle='-')
plt.plot(NH4_data['time'], NH4_data['nh4']*1000,
         label='NH4 (microgram/l)', color='red', linestyle='-')
plt.xlabel('Time'); plt.ylabel('Values')
plt.xticks(rotation=45); plt.legend(); plt.show()

# ---- Multi-subplot (raw data) ----
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(flowdata['time'], flowdata['flow'], color='blue', linestyle="-", label="Flow (m3/s)")
ax1.set_ylabel('Flow [m3/hr]'); ax1.legend(); ax1.grid(True)

ax2 = plt.subplot(3, 1, 2)
ax2.plot(NH4_data['time'], NH4_data['nh4'], color='red', linestyle="-", label="NH4 (mg/l)")
ax2.set_ylabel('NH4 [mg/l]'); ax2.legend(); ax2.grid(True)

ax3 = plt.subplot(3, 1, 3)
ax3.plot(NH4_data['time'], NH4_flux*1e-3, color='magenta', linestyle="-", label="Flux (kg/hr)")
ax3.set_ylabel('NH4 load [kg/hr]'); ax3.legend(); ax3.grid(True)

xt = pd.date_range(start=min(flowdata['time']), end=max(flowdata['time']), freq="48h")
ax1.set_xticks(xt); ax1.set_xticklabels([])
ax2.set_xticks(xt); ax2.set_xticklabels([])
ax3.set_xticks(xt); ax3.set_xticklabels(xt.strftime("%m-%d"), rotation=45)
plt.tight_layout(); plt.show()

# ---- Interpolation & smoothing ----
flowdata['flow'] = flowdata['flow'].interpolate(method='linear')
flowdata = flowdata.drop(flowdata.index[0:29]).reset_index(drop=True)
flowdata['smoothed'] = flowdata['flow'].rolling(window=60).mean().fillna(flowdata['flow'])

NH4_data['nh4'] = NH4_data['nh4'].interpolate(method='linear')
NH4_data = NH4_data.drop(flowdata.index[0:29]).reset_index(drop=True)
NH4_data['smoothed'] = NH4_data['nh4'].rolling(window=60).mean().fillna(NH4_data['nh4'])

NH4_flux_smoothed = flowdata['smoothed'] * NH4_data['smoothed']   # g/hr

# ---- Plot smoothed data ----
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(flowdata['time'], flowdata['smoothed'], color='blue', linestyle="-", label="Flow (m3/s)")
ax1.set_ylabel('Flow [m3/hr]'); ax1.legend(); ax1.grid(True)

ax2 = plt.subplot(3, 1, 2)
ax2.plot(NH4_data['time'], NH4_data['smoothed'], color='red', linestyle="-", label="NH4 (mg/l)")
ax2.set_ylabel('NH4 [mg/l]'); ax2.legend(); ax2.grid(True)

ax3 = plt.subplot(3, 1, 3)
ax3.plot(NH4_data['time'], NH4_flux_smoothed*1e-3, color='magenta', linestyle="-", label="Flux (kg/hr)")
ax3.set_ylabel('NH4 load [kg/hr]'); ax3.legend(); ax3.grid(True)

xt = pd.date_range(start=min(flowdata['time']), end=max(flowdata['time']), freq="8h")
ax1.set_xticks(xt); ax1.set_xticklabels([])
ax2.set_xticks(xt); ax2.set_xticklabels([])
ax3.set_xticks(xt); ax3.set_xticklabels(xt.strftime("%m-%d-%H"), rotation=90)

zoom_start, zoom_end = pd.Timestamp('2018-06-08'), pd.Timestamp('2018-06-15')
ax1.set_xlim(zoom_start, zoom_end)
ax2.set_xlim(zoom_start, zoom_end)
ax3.set_xlim(zoom_start, zoom_end)
plt.tight_layout(); plt.show()

# ---- Model 0 ----
data = flowdata.merge(NH4_data[['time', 'nh4', 'smoothed']], on='time')
data['NH4_load'] = NH4_flux_smoothed
dry = data[flowdata['smoothed'] < 5000].copy()
time_idx = pd.DatetimeIndex(dry['time'])
t = time_idx.hour/24 + time_idx.minute/1440
X = np.column_stack([
    np.ones(len(t)),
    np.sin(2*np.pi*t),
    np.cos(2*np.pi*t),
    np.sin(4*np.pi*t),
    np.cos(4*np.pi*t)
])
y = dry['NH4_load']
params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print("Estimated parameters [a0, a1, b1, a2, b2]:"); print(params)
data['flow'] = data['smoothed_x']
Model_results = NH4inletModel0(params, data)

# ---- Plot Model 0 results ----
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(NH4_data['time'], NH4_data['smoothed'], color='blue', linestyle="-", label="Concentration (g/m3)")
ax1.plot(Model_results['time'], Model_results['simNH4conc'], color='green', linestyle="-", label="Simulated concentration")
ax1.set_ylabel('NH4 [g/m3]'); ax1.legend(); ax1.grid(True); ax1.set_ylim([0, 100])

ax2 = plt.subplot(2, 1, 2)
ax2.plot(NH4_data['time'], NH4_flux_smoothed, color='red', linestyle="-", label="NH4 (g/hr)")
ax2.plot(Model_results['time'], Model_results['simNH4load'], linestyle="-", color="green", label="Simulated NH4 load")
ax2.set_ylabel('NH4 load [g/hr]'); ax2.legend(); ax2.grid(True)

xt = pd.date_range(start=min(flowdata['time']), end=max(flowdata['time']), freq="24h")
ax1.set_xticks(xt); ax1.set_xticklabels([])
ax2.set_xticks(xt); ax2.set_xticklabels(xt.strftime("%m-%d-%H"), rotation=90)
plt.tight_layout(); plt.show()

# ---- Model 1 ----
[a0, a1, b1, a2, b2] = params
param_model1 = [a0, a1, b1, a2, b2, 11500]
model1_results = NH4inletModel1(param_model1, data)

# ---- Plot Model 1 results ----
plt.figure(figsize=(10, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(NH4_data['time'], NH4_data['smoothed'], color='blue', linestyle="-", label="Concentration (g/m3)")
ax1.plot(model1_results['time'], model1_results['simNH4conc'], color='green', linestyle="-", label="Simulated concentration")
ax1.set_ylabel('NH4 [g/m3]'); ax1.legend(); ax1.grid(True); ax1.set_ylim([0, 100])

ax2 = plt.subplot(2, 1, 2)
ax2.plot(NH4_data['time'], NH4_flux_smoothed, color='red', linestyle="-", label="NH4 (g/hr)")
ax2.plot(model1_results['time'], model1_results['simNH4load'], linestyle="-", color="green", label="Simulated NH4 load")
ax2.set_ylabel('NH4 load [g/hr]'); ax2.legend(); ax2.grid(True)

xt = pd.date_range(start=min(flowdata['time']), end=max(flowdata['time']), freq="24h")
ax1.set_xticks(xt); ax1.set_xticklabels([])
ax2.set_xticks(xt); ax2.set_xticklabels(xt.strftime("%m-%d-%H"), rotation=90)
plt.tight_layout(); plt.show()

# ---- Model 3 ----
param_model3 = [a0, a1, b1, a2, b2, 0.1, 2, 0.1]
model3_results = NH4inletModel3(param_model3, data, 5000)

# ---- Plot Model 3 results ----
plt.figure(figsize=(10, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(NH4_data['time'], NH4_data['smoothed'], color='blue', linestyle="-", label="Concentration (g/m3)")
ax1.plot(model3_results['time'], model3_results['simNH4conc'], color='green', linestyle="-", label="Simulated concentration")
ax1.set_ylabel('NH4 [g/m3]'); ax1.legend(); ax1.grid(True); ax1.set_ylim([0, 100])

ax2 = plt.subplot(2, 1, 2)
ax2.plot(NH4_data['time'], NH4_flux_smoothed, color='red', linestyle="-", label="NH4 (g/hr)")
ax2.plot(model3_results['time'], model3_results['simNH4load'], linestyle="-", color="green", label="Simulated NH4 load")
ax2.set_ylabel('NH4 load [g/hr]'); ax2.legend(); ax2.grid(True)

xt = pd.date_range(start=min(flowdata['time']), end=max(flowdata['time']), freq="24h")
ax1.set_xticks(xt); ax1.set_xticklabels([])
ax2.set_xticks(xt); ax2.set_xticklabels(xt.strftime("%m-%d-%H"), rotation=90)
plt.tight_layout(); plt.show()

# ---- Model 4 ----
param_model4 = [a0, a1, b1, a2, b2, 0.1, 2, 0.1, 11500]
model4_results = NH4inletModel4(param_model4, data, 5000)

# ---- Plot Model 4 results ----
plt.figure(figsize=(10, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(NH4_data['time'], NH4_data['smoothed'], color='blue', linestyle="-", label="Concentration (g/m3)")
ax1.plot(model4_results['time'], model4_results['simNH4conc'], color='green', linestyle="-", label="Simulated concentration")
ax1.set_ylabel('NH4 [g/m3]'); ax1.legend(); ax1.grid(True); ax1.set_ylim([0, 100])

ax2 = plt.subplot(2, 1, 2)
ax2.plot(NH4_data['time'], NH4_flux_smoothed, color='red', linestyle="-", label="NH4 (g/hr)")
ax2.plot(model4_results['time'], model4_results['simNH4load'], linestyle="-", color="green", label="Simulated NH4 load")
ax2.set_ylabel('NH4 load [g/hr]'); ax2.legend(); ax2.grid(True)

xt = pd.date_range(start=min(flowdata['time']), end=max(flowdata['time']), freq="24h")
ax1.set_xticks(xt); ax1.set_xticklabels([])
ax2.set_xticks(xt); ax2.set_xticklabels(xt.strftime("%m-%d-%H"), rotation=90)
plt.tight_layout(); plt.show()

# -*- coding: utf-8 -*-

"""
Task 2.1 — Local Sensitivity Analysis (OAT)
"""

# Setup
import os
os.chdir(r"C:\Users\rraat\OneDrive - Danmarks Tekniske Universitet\Documents\Academic\DTU\1ST SEMESTER\12104 Environmental Modelling\Module 4\Assignment 4")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from NH4models import NH4inletModel0, NH4inletModel1, NH4inletModel3, NH4inletModel4
import objective_functions as objfun  # <-- your custom metrics

# Load and preprocess data
flowdata = pd.read_csv('flowdata_DAM_180604_180715.csv', sep=';', parse_dates=['time'])
NH4_data = pd.read_csv('NH4data_180604_180715.csv', sep=';', parse_dates=['time'])

flowdata['flow'] = flowdata['flow'].interpolate(method='linear')
NH4_data['nh4'] = NH4_data['nh4'].interpolate(method='linear')

flowdata = flowdata.drop(flowdata.index[0:29]).reset_index(drop=True)
NH4_data = NH4_data.drop(flowdata.index[0:29]).reset_index(drop=True)

flowdata['smoothed'] = flowdata['flow'].rolling(window=60).mean().fillna(flowdata['flow'])
NH4_data['smoothed'] = NH4_data['nh4'].rolling(window=60).mean().fillna(NH4_data['nh4'])

NH4_flux_smoothed = flowdata['smoothed'] * NH4_data['smoothed']  # g/hr

# Merge and prepare model input
data = flowdata.merge(NH4_data[['time', 'nh4', 'smoothed']], on='time')
data['NH4_load'] = NH4_flux_smoothed
data['flow'] = data['smoothed_x']

# Estimate Fourier coefficients (Model 0)
dry = data[data['flow'] < 5000].copy()
t_idx = pd.DatetimeIndex(dry['time'])
t = t_idx.hour / 24 + t_idx.minute / 1440
X = np.column_stack([
    np.ones(len(t)),
    np.sin(2*np.pi*t),
    np.cos(2*np.pi*t),
    np.sin(4*np.pi*t),
    np.cos(4*np.pi*t)
])
y = dry['NH4_load']
params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
[a0, a1, b1, a2, b2] = params

# Select model and baseline parameters
MODEL = 'M4'
baseline = [a0, a1, b1, a2, b2, 0.1, 2, 0.1, 11500]
param_names = ['a0','a1','b1','a2','b2','lambda1','Y2','Y3','V']
Qthr = 5000
OBJ_FUN = objfun.NSE  # <- use RMSE, MARE, invMSE, or NSE

# Run baseline simulation
def run_model(model, params, df, Qthr):
    if model == 'M0':
        return NH4inletModel0(params, df)
    elif model == 'M1':
        return NH4inletModel1(params, df)
    elif model == 'M3':
        return NH4inletModel3(params, df, Qthr)
    elif model == 'M4':
        return NH4inletModel4(params, df, Qthr)

baseline_out = run_model(MODEL, baseline, data, Qthr)
y_obs = NH4_data['smoothed']
y_sim = baseline_out['simNH4conc']
J0 = OBJ_FUN(y_obs, y_sim)

# ---- OAT ±10% per parameter ----
delta = 0.10
results = []

for i, pname in enumerate(param_names):
    p_plus = copy.deepcopy(baseline)
    p_minus = copy.deepcopy(baseline)

    bump = delta * baseline[i] if baseline[i] != 0 else delta
    p_plus[i] += bump
    p_minus[i] -= bump

    out_plus = run_model(MODEL, p_plus, data, Qthr)
    out_minus = run_model(MODEL, p_minus, data, Qthr)

    J_plus = OBJ_FUN(y_obs, out_plus['simNH4conc'])
    J_minus = OBJ_FUN(y_obs, out_minus['simNH4conc'])

    SI = (J_plus - J_minus) / (2 * delta * (J0 if J0 != 0 else 1))
    asym = np.abs(J_plus - J0) - np.abs(J_minus - J0)

    results.append({
        'param': pname,
        'baseline': baseline[i],
        'J0': J0,
        'J+10%': J_plus,
        'J-10%': J_minus,
        'SI_norm': SI,
        'asymmetry': asym
    })

# ---- Output table ----
df_results = pd.DataFrame(results)
print("\n=== Task 2.1 OAT Sensitivity Results ===")
print(df_results[['param','baseline','SI_norm','asymmetry','J0','J+10%','J-10%']].round(4))

# ---- Plot sensitivity indices ----
plt.figure(figsize=(8, 4))
plt.bar(df_results['param'], df_results['SI_norm'], color=['#4c72b0' if v>=0 else '#dd8452' for v in df_results['SI_norm']])
plt.axhline(0, color='black', linewidth=1)
plt.ylabel('Normalized Sensitivity Index')
plt.title(f'OAT Sensitivity — Model {MODEL} — Metric: {OBJ_FUN.__name__}')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()