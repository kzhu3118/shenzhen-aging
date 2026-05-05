import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "pop_census" / "city_census_panel_2000_2010_2020.xlsx"
PROJ_FILE = BASE_DIR / "data" / "processed" / "city_age_projection_2020_2050.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "city_age_panel_2000_2050_calibrated.xlsx"


def normalize_city_name(name):
    name = str(name).strip()
    if name.endswith("市"):
        name = name[:-1]
    return name


def safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return num / den


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find census file: {DATA_FILE}")
    if not PROJ_FILE.exists():
        raise FileNotFoundError(f"Cannot find projection file: {PROJ_FILE}")

    # =========================
    # 1. Read census data.
    # =========================
    census = pd.read_excel(DATA_FILE)
    census = census.copy()
    census["city"] = census["city"].astype(str).str.strip()
    census["city_short"] = census["city"].apply(normalize_city_name)

    # Standardize column names.
    census = census.rename(
        columns={
            "pop_0_14": "child",
            "pop_15_59": "labor",
            "pop_60_plus": "old",
            "total_pop": "total",
            "aging_rate_60_plus": "aging_rate",
        }
    )

    census_keep = ["city", "city_short", "year", "child", "labor", "old", "total", "aging_rate"]
    census = census[census_keep].copy()

    # =========================
    # 2. Read WorldPop projection data.
    # =========================
    proj = pd.read_excel(PROJ_FILE)
    proj = proj.copy()
    proj["city"] = proj["市"].astype(str).str.strip()
    proj["city_short"] = proj["city"].apply(normalize_city_name)

    proj = proj.rename(
        columns={
            "pop_child": "child",
            "pop_labor": "labor",
            "pop_old": "old",
            "pop_total_selected": "total",
            "share_old": "aging_rate",
        }
    )

    proj_keep = [
        "省",
        "省代码",
        "市代码",
        "city",
        "city_short",
        "year",
        "child",
        "labor",
        "old",
        "total",
        "aging_rate",
    ]
    proj = proj[proj_keep].copy()

    # =========================
    # 3. Build 2020 calibration factors.
    # =========================
    census_2020 = census[census["year"] == 2020].copy()
    proj_2020 = proj[proj["year"] == 2020].copy()

    calib = pd.merge(
        census_2020[["city_short", "child", "labor", "old"]],
        proj_2020[["city_short", "child", "labor", "old"]],
        on="city_short",
        how="inner",
        suffixes=("_census", "_wp"),
    )

    calib["child_factor"] = calib.apply(lambda x: safe_ratio(x["child_census"], x["child_wp"]), axis=1)
    calib["labor_factor"] = calib.apply(lambda x: safe_ratio(x["labor_census"], x["labor_wp"]), axis=1)
    calib["old_factor"] = calib.apply(lambda x: safe_ratio(x["old_census"], x["old_wp"]), axis=1)

    calib = calib[["city_short", "child_factor", "labor_factor", "old_factor"]].copy()

    # =========================
    # 4. Calibrate WorldPop 2020-2050 values.
    # =========================
    proj_adj = pd.merge(proj, calib, on="city_short", how="left")

    proj_adj["child_calibrated"] = proj_adj["child"] * proj_adj["child_factor"]
    proj_adj["labor_calibrated"] = proj_adj["labor"] * proj_adj["labor_factor"]
    proj_adj["old_calibrated"] = proj_adj["old"] * proj_adj["old_factor"]

    # Fall back to original values when no factor is matched.
    proj_adj["child_calibrated"] = proj_adj["child_calibrated"].fillna(proj_adj["child"])
    proj_adj["labor_calibrated"] = proj_adj["labor_calibrated"].fillna(proj_adj["labor"])
    proj_adj["old_calibrated"] = proj_adj["old_calibrated"].fillna(proj_adj["old"])

    proj_adj["total_calibrated"] = (
        proj_adj["child_calibrated"] + proj_adj["labor_calibrated"] + proj_adj["old_calibrated"]
    )
    proj_adj["aging_rate_calibrated"] = proj_adj["old_calibrated"] / proj_adj["total_calibrated"]

    # Replace 2020 values directly with census values.
    census_2020_for_replace = census_2020[["city_short", "child", "labor", "old", "total", "aging_rate"]].rename(
        columns={
            "child": "child_census",
            "labor": "labor_census",
            "old": "old_census",
            "total": "total_census",
            "aging_rate": "aging_rate_census",
        }
    )

    proj_adj = pd.merge(proj_adj, census_2020_for_replace, on="city_short", how="left")

    is_2020 = proj_adj["year"] == 2020
    proj_adj.loc[is_2020, "child_calibrated"] = proj_adj.loc[is_2020, "child_census"]
    proj_adj.loc[is_2020, "labor_calibrated"] = proj_adj.loc[is_2020, "labor_census"]
    proj_adj.loc[is_2020, "old_calibrated"] = proj_adj.loc[is_2020, "old_census"]
    proj_adj.loc[is_2020, "total_calibrated"] = proj_adj.loc[is_2020, "total_census"]
    proj_adj.loc[is_2020, "aging_rate_calibrated"] = proj_adj.loc[is_2020, "aging_rate_census"]

    # Keep future projection years for the final projection panel.
    proj_future = proj_adj[proj_adj["year"].isin([2025, 2030, 2035, 2040, 2045, 2050])].copy()

    proj_future_out = pd.DataFrame({
        "city": proj_future["city"],
        "city_short": proj_future["city_short"],
        "year": proj_future["year"],
        "child": proj_future["child_calibrated"],
        "labor": proj_future["labor_calibrated"],
        "old": proj_future["old_calibrated"],
        "total": proj_future["total_calibrated"],
        "aging_rate": proj_future["aging_rate_calibrated"],
        "source": "worldpop_calibrated_by_census2020",
    })

    # =========================
    # 5. Prepare the census portion.
    # =========================
    census_out = census.copy()
    census_out["source"] = "census"

    # =========================
    # 6. Merge into the final panel.
    # =========================
    final_df = pd.concat([census_out, proj_future_out], ignore_index=True)
    final_df = final_df.sort_values(["city_short", "year"]).reset_index(drop=True)

    # Add analysis-ready population columns.
    final_df["pop_0_14"] = final_df["child"]
    final_df["pop_15_59"] = final_df["labor"]
    final_df["pop_60_plus"] = final_df["old"]
    final_df["total_pop"] = final_df["total"]
    final_df["aging_rate_60_plus"] = final_df["aging_rate"]

    final_df = final_df[
        [
            "city",
            "city_short",
            "year",
            "pop_0_14",
            "pop_15_59",
            "pop_60_plus",
            "total_pop",
            "aging_rate_60_plus",
            "source",
        ]
    ].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_excel(OUTPUT_FILE, index=False)

    print("-" * 40)
    print("2000-2050 city age-structure panel generated.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total rows: {len(final_df)}")
    print(f"City count: {final_df['city_short'].nunique()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Merge and calibration failed:")
        traceback.print_exc()
