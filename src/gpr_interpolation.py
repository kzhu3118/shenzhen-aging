import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, ConstantKernel, WhiteKernel)
from joblib import Parallel, delayed
import multiprocessing
import warnings
import time
from pathlib import Path

# --- 0. Initial setup ---
warnings.filterwarnings('ignore')
BASE_DIR = Path(__file__).resolve().parents[1]

# Use all available CPU cores except one reserved for the system.
N_CORES = max(1, multiprocessing.cpu_count() - 1)
print(f"Detected {multiprocessing.cpu_count()} CPU cores; using {N_CORES} cores for parallel processing.")

if 'pop_predict' not in globals():
    pop_predict = gpd.read_file(BASE_DIR / 'data' / 'raw' / 'pop_predict.gpkg')
pop_predict['pop_2020'] = pop_predict[['child_2020_sum','labor_2020_sum','old_2020_sum']].sum(axis=1)
pop_predict_control = pop_predict[['Id','pop_2020','geometry']].copy()

china_population = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'world_population.csv')

# --- 1 & 2. Calculate China's 2020 baseline age-structure proportions ---

print("--- Tasks 1 & 2: Calculate China's baseline age structure ---")

UN_AGE_COLS = [
    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
    '80-84', '85-89', '90-94', '95-99', '100+'
]
UN_TOTAL_POP_COL = 'Total pop'

try:
    china_population['Year'] = pd.to_numeric(china_population['Year'], errors='coerce')
    cols_to_clean = UN_AGE_COLS + [UN_TOTAL_POP_COL]
    for col in cols_to_clean:
        if col in china_population.columns:
            china_population[col] = china_population[col].astype(str).str.replace(r'[,\s]', '', regex=True)
            china_population[col] = pd.to_numeric(china_population[col], errors='coerce')
    print("China population data cleaned.")
except Exception as e:
    print(f"Data cleaning failed: {e}. Check the CSV format.")

china_pop_2020_row = china_population[china_population['Year'] == 2020].iloc[0]

eighty_plus_cols = ['80-84', '85-89', '90-94', '95-99', '100+']
existing_80plus_cols = [col for col in eighty_plus_cols if col in china_pop_2020_row.index]
china_pop_2020_row['pop_80_plus'] = china_pop_2020_row[existing_80plus_cols].sum()

SCCM_AGE_GROUPS_CHINA = [
    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', 'pop_80_plus'
]
total_pop_china = china_pop_2020_row[UN_TOTAL_POP_COL]

china_proportions = {}
for col in SCCM_AGE_GROUPS_CHINA:
    china_proportions[col] = china_pop_2020_row[col] / total_pop_china

print("China age-structure proportions (S_Control):")
print({k: f"{v*100:.2f}%" for k, v in china_proportions.items()})


# --- 3. Apply the baseline structure to Shenzhen grids and run SCCM ---

print("\n--- Task 3: Apply S_Control to Shenzhen and run SCCM ---")

prop_to_sccm_map = {
    '45-49': 'pop_45',
    '50-54': 'pop_50',
    '55-59': 'pop_55',
    '60-64': 'pop_60',
    '65-69': 'pop_65',
    '70-74': 'pop_70',
    '75-79': 'pop_75',
    'pop_80_plus': 'pop_80'
}

gdf = pop_predict_control.copy()

for china_col, proportion in china_proportions.items():
    sccm_col = prop_to_sccm_map[china_col]
    gdf[sccm_col] = gdf['pop_2020'] * proportion

print("S_Control 2020 grid population structure generated.")

cohort_2020 = ['pop_60', 'pop_65', 'pop_70', 'pop_75', 'pop_80']
cohort_2025 = ['pop_55', 'pop_60', 'pop_65', 'pop_70', 'pop_75']
cohort_2030 = ['pop_50', 'pop_55', 'pop_60', 'pop_65', 'pop_70']
cohort_2035 = ['pop_45', 'pop_50', 'pop_55', 'pop_60', 'pop_65']

gdf['control_old_2020_sum'] = gdf[cohort_2020].sum(axis=1)
gdf['control_old_2025_sum'] = gdf[cohort_2025].sum(axis=1)
gdf['control_old_2030_sum'] = gdf[cohort_2030].sum(axis=1)
gdf['control_old_2035_sum'] = gdf[cohort_2035].sum(axis=1)

print("S_Control anchor-year older-adult population calculated for 2020, 2025, 2030, and 2035.")


# --- 4. Use GPR to interpolate annual older-adult population for 2020-2035 ---

print("\n--- Task 4: Run Gaussian Process Regression interpolation in parallel ---")

# GPR settings
KNOWN_YEARS = np.array([2020, 2025, 2030, 2035])
TARGET_YEARS = np.arange(2020, 2036)
ANCHOR_COLS = ['control_old_2020_sum', 'control_old_2025_sum', 'control_old_2030_sum', 'control_old_2035_sum']

# GPR kernel
kernel = ConstantKernel(1.0) * Matern(length_scale=5.0, nu=2.5) + WhiteKernel(noise_level=1.0)


# Define row-level GPR work as a top-level function for multiprocessing.
def gpr_interpolate_row(data_points, known_years, target_years, kernel):
    """Interpolate one grid cell with GPR."""
    y = np.array(data_points).astype(float)
    X = known_years.reshape(-1, 1)

    valid_mask = ~np.isnan(y)
    if np.sum(valid_mask) < 2:
        return np.full(len(target_years), np.nan)

    X_train = X[valid_mask]
    y_train = y[valid_mask]
    X_predict = target_years.reshape(-1, 1)

    try:
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        gpr.fit(X_train, y_train)
        y_pred, sigma = gpr.predict(X_predict, return_std=True)
        y_pred[y_pred < 0] = 0
        return y_pred
    except Exception:
        return np.full(len(target_years), np.nan)


# Extract anchor data as a NumPy array to avoid passing the full GeoDataFrame.
anchor_data = gdf[ANCHOR_COLS].values  # shape: (n_grids, 4)
n_grids = len(anchor_data)

print(f"Interpolating {n_grids} grids with {N_CORES} parallel workers...")

start_time = time.time()

# Run GPR interpolation in parallel with joblib.
results = Parallel(
    n_jobs=N_CORES,
    backend='loky',
    verbose=10,
)(
    delayed(gpr_interpolate_row)(anchor_data[i], KNOWN_YEARS, TARGET_YEARS, kernel)
    for i in range(len(anchor_data))
)

elapsed = time.time() - start_time
print(f"\nS_Control interpolation complete. Elapsed: {elapsed:.1f} seconds using {N_CORES} cores.")

# Format outputs and join them back to the GeoDataFrame.
new_cols = [f'control_old_{year}_gpr' for year in TARGET_YEARS]
interpolated_df = pd.DataFrame(
    results,
    columns=new_cols,
    index=gdf.index
)

gdf_final_control = gdf.join(interpolated_df)


# --- 5. Final results ---
print("\n--- Simulation complete ---")
print("S_Control annual older-adult population has been generated and joined.")

cols_to_show = ['Id', 'pop_2020', 'geometry',
                'control_old_2020_sum', 'control_old_2035_sum',
                'control_old_2020_gpr', 'control_old_2021_gpr',
                'control_old_2035_gpr']
cols_to_show = [col for col in cols_to_show if col in gdf_final_control.columns]
print(gdf_final_control[cols_to_show].head())
