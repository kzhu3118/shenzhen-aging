import pandas as pd
from pathlib import Path
import traceback

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "pop_census"

INPUT_FILES = {
    2000: DATA_DIR / "merged_national_city_age_2000.xlsx",
    2010: DATA_DIR / "merged_national_city_age_2010.xlsx",
    2020: DATA_DIR / "merged_national_city_age_2020.xlsx",
}

OUTPUT_FILE = DATA_DIR / "city_census_panel_2000_2010_2020.xlsx"

AGE_0_14 = ["0岁", "1-4岁", "5-9岁", "10-14岁"]
AGE_15_59 = ["15-19岁", "20-24岁", "25-29岁", "30-34岁", "35-39岁", "40-44岁", "45-49岁", "50-54岁", "55-59岁"]
AGE_60_PLUS = ["60-64岁", "65-69岁", "70-74岁", "75-79岁", "80-84岁", "85岁及以上"]


def build_age_cols(age_groups):
    cols = []
    for group in age_groups:
        cols.extend([f"{group}_男", f"{group}_女"])
    return cols


def validate_columns(df, required_cols, year):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{year} is missing columns: {missing}")


def process_year_file(year, file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    print(f"Processing {year}: {file_path.name}")
    df = pd.read_excel(file_path)
    df["地区"] = df["地区"].astype(str).str.replace(r"\s+", "", regex=True)

    age_cols_0_14 = build_age_cols(AGE_0_14)
    age_cols_15_59 = build_age_cols(AGE_15_59)
    age_cols_60_plus = build_age_cols(AGE_60_PLUS)
    all_age_cols = age_cols_0_14 + age_cols_15_59 + age_cols_60_plus

    validate_columns(df, ["地区"] + all_age_cols, year)

    for col in all_age_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    panel_df = pd.DataFrame()
    panel_df["city"] = df["地区"]
    panel_df["year"] = year
    panel_df["pop_0_14"] = df[age_cols_0_14].sum(axis=1)
    panel_df["pop_15_59"] = df[age_cols_15_59].sum(axis=1)
    panel_df["pop_60_plus"] = df[age_cols_60_plus].sum(axis=1)
    panel_df["total_pop"] = panel_df["pop_0_14"] + panel_df["pop_15_59"] + panel_df["pop_60_plus"]
    panel_df["aging_rate_60_plus"] = panel_df["pop_60_plus"] / panel_df["total_pop"]

    return panel_df


def merge_census_panel():
    frames = []
    for year, file_path in INPUT_FILES.items():
        frames.append(process_year_file(year, file_path))

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.sort_values(["city", "year"]).reset_index(drop=True)

    return final_df


if __name__ == "__main__":
    try:
        final_panel = merge_census_panel()
        final_panel.to_excel(OUTPUT_FILE, index=False)

        print("-" * 30)
        print("Three-period census panel generated.")
        print(f"Total rows: {len(final_panel)}")
        print(f"City count: {final_panel['city'].nunique()}")
        print(f"Output file: {OUTPUT_FILE}")

    except Exception:
        print("Merge failed:")
        traceback.print_exc()
