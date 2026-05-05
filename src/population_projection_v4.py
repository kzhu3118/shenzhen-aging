import traceback
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "china_city_worldpop_age_2020_wide.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "city_age_projection_2020_2050.xlsx"

YEARS = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

AGE_SETS = {
    "child": {
        2020: [0, 1, 5, 10],
        2025: [0, 1, 5, 10],
        2030: [0, 1, 5, 10],
        2035: [0, 1, 5, 10],
        2040: [0, 1, 5, 10],
        2045: [0, 1, 5, 10],
        2050: [0, 1, 5, 10],
    },
    "labor": {
        2020: [15, 20, 25, 30, 35, 40, 45, 50, 55],
        2025: [10, 15, 20, 25, 30, 35, 40, 45, 50],
        2030: [5, 10, 15, 20, 25, 30, 35, 40, 45],
        2035: [0, 5, 10, 15, 20, 25, 30, 35, 40],
        2040: [0, 1, 5, 10, 15, 20, 25, 30, 35],
        2045: [0, 1, 5, 10, 15, 20, 25, 30],
        2050: [0, 1, 5, 10, 15, 20, 25],
    },
    "old": {
        2020: [60, 65, 70, 75, 80],
        2025: [55, 60, 65, 70, 75],
        2030: [50, 55, 60, 65, 70],
        2035: [45, 50, 55, 60, 65],
        2040: [40, 45, 50, 55, 60],
        2045: [35, 40, 45, 50, 55],
        2050: [30, 35, 40, 45, 50],
    },
}

SEXES = ["female", "male"]
META_COLS = ["city_id", "省", "省代码", "市", "市代码", "市类型", "省类型"]


def age_label_from_start(age_start):
    if age_start == 0:
        return "0"
    if age_start == 1:
        return "1-4"
    if age_start == 80:
        return "80+"
    return f"{age_start}-{age_start + 4}"


def build_age_col(age_start, sex):
    return f"{sex}_{age_label_from_start(age_start)}"


def validate_columns(df):
    required = set(META_COLS)
    for sex in SEXES:
        for age_start in [0, 1] + list(range(5, 85, 5)):
            required.add(build_age_col(age_start, sex))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")


def sum_age_group(df, age_starts):
    cols = []
    for sex in SEXES:
        for age_start in age_starts:
            cols.append(build_age_col(age_start, sex))
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].sum(axis=1)


def project_age_structure():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_FILE)
    validate_columns(df)

    result_frames = []
    for year in YEARS:
        temp = df[META_COLS].copy()
        temp["year"] = year
        temp["pop_child"] = sum_age_group(df, AGE_SETS["child"][year])
        temp["pop_labor"] = sum_age_group(df, AGE_SETS["labor"][year])
        temp["pop_old"] = sum_age_group(df, AGE_SETS["old"][year])
        temp["pop_total_selected"] = temp["pop_child"] + temp["pop_labor"] + temp["pop_old"]
        temp["share_child"] = temp["pop_child"] / temp["pop_total_selected"]
        temp["share_labor"] = temp["pop_labor"] / temp["pop_total_selected"]
        temp["share_old"] = temp["pop_old"] / temp["pop_total_selected"]
        result_frames.append(temp)

    final_df = pd.concat(result_frames, ignore_index=True)
    final_df = final_df.sort_values(["市代码", "year"]).reset_index(drop=True)
    return final_df


if __name__ == "__main__":
    try:
        final_df = project_age_structure()
        final_df.to_excel(OUTPUT_FILE, index=False)
        print("2020-2050 city age-structure projection complete.")
        print(f"Output file: {OUTPUT_FILE}")
    except Exception:
        print("Age-structure projection failed:")
        traceback.print_exc()
