import pandas as pd
import traceback
from pathlib import Path

# 1. Set paths.
BASE_DIR = Path(__file__).resolve().parents[1]
file_path = BASE_DIR / "data" / "raw" / "pop_census" / "2000_pop_census.xlsx"
output_path = BASE_DIR / "data" / "processed" / "pop_census" / "merged_national_city_age_2000.xlsx"

# 2. Define age groups.
age_groups_p1 = [
    "0岁",
    "1-4岁",
    "5-9岁",
    "10-14岁",
    "15-19岁",
    "20-24岁",
    "25-29岁",
    "30-34岁",
    "35-39岁",
    "40-44岁",
]
age_groups_p2 = [
    "45-49岁",
    "50-54岁",
    "55-59岁",
    "60-64岁",
    "65-69岁",
    "70-74岁",
    "75-79岁",
    "80-84岁",
    "85岁及以上",
]


def parse_block(df_block):
    """
    Parse one age block.

    The 2000 census table repeats headers and sometimes splits region names
    across two rows.
    """
    rows = []

    for _, row in df_block.iterrows():
        region = row.iloc[0]
        if pd.isna(region):
            continue

        region = str(region)
        if any(keyword in region for keyword in ["县市名", "单位", "表2分年龄"]):
            continue

        values = row.iloc[1:]
        if values.notna().sum() == 0:
            if rows:
                rows[-1][0] += region.strip()
            continue

        rows.append([region] + values.tolist())

    return pd.DataFrame(rows)


def process_2000_census(file_path):
    print("Reading 2000 census data...")
    df = pd.read_excel(file_path, sheet_name="1-98", header=None)

    # Left block: ages 0-44, starting at row 9.
    left_raw = df.iloc[8:, 0:21].copy()
    left_df = parse_block(left_raw)
    left_df.columns = ["地区_raw"] + [f"{g}_{sex}" for g in age_groups_p1 for sex in ["男", "女"]]

    # Right block: ages 45+, starting at row 7.
    right_raw = df.iloc[6:, 23:42].copy()
    right_df = parse_block(right_raw)
    right_df.columns = ["地区_raw_right"] + [f"{g}_{sex}" for g in age_groups_p2 for sex in ["男", "女"]]

    if len(left_df) != len(right_df):
        raise ValueError(f"Left and right blocks have different row counts: left={len(left_df)}, right={len(right_df)}")

    df_combined = pd.concat(
        [left_df.reset_index(drop=True), right_df.drop(columns=["地区_raw_right"]).reset_index(drop=True)],
        axis=1,
    )

    # Keep original indentation for hierarchy detection and clean region names.
    df_combined["地区"] = df_combined["地区_raw"].astype(str).str.replace(r"\s+", "", regex=True)
    df_combined["indent"] = df_combined["地区_raw"].astype(str).str.extract(r"^(\s*)", expand=False).str.len()

    data_cols = [c for c in df_combined.columns if c not in ["地区_raw", "地区", "indent"]]
    for col in data_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

    # Use indentation to filter municipality-level and prefecture-level cities.
    df_city = df_combined[
        df_combined["地区"].str.endswith("市")
        & ~df_combined["地区"].str.contains("合计|省|自治区|特别行政区|市辖区")
        & df_combined["indent"].isin([0, 2])
    ].copy()

    # Drop helper columns.
    df_city.drop(columns=["地区_raw", "indent"], inplace=True)

    return df_city


if __name__ == "__main__":
    try:
        final_df_2000 = process_2000_census(file_path)

        older_cols = [
            c
            for c in final_df_2000.columns
            if any(x in c for x in ["60-64", "65-69", "70-74", "75-79", "80-84", "85岁"])
        ]
        final_df_2000["2000_60Plus_Total"] = final_df_2000[older_cols].sum(axis=1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df_2000.to_excel(output_path, index=False)
        print("-" * 30)
        print("2000 census processing complete.")
        print(f"Filtered city count: {len(final_df_2000)}")
        print(f"Saved to: {output_path}")

    except Exception:
        print("Run failed:")
        traceback.print_exc()
