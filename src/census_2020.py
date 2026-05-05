import pandas as pd
import numpy as np
from pathlib import Path

# 1. Set paths.
BASE_DIR = Path(__file__).resolve().parents[1]
file_path = BASE_DIR / 'data' / 'raw' / 'pop_census' / '2020_pop_census.xlsx'
output_path = BASE_DIR / 'data' / 'processed' / 'pop_census' / 'merged_national_city_age_2020.xlsx'

# 2. Define age groups in the same order as the Excel file.
age_groups_p1 = ['0岁', '1-4岁', '5-9岁', '10-14岁', '15-19岁', '20-24岁', '25-29岁', '30-34岁', '35-39岁', '40-44岁']
age_groups_p2 = ['45-49岁', '50-54岁', '55-59岁', '60-64岁', '65-69岁', '70-74岁', '75-79岁', '80-84岁', '85岁及以上']

def process_2020_census(file_path):
    print("Reading data...")
    df = pd.read_excel(file_path, header=None)
    
    # Data usually starts at row 6 after the total row.
    data_start_idx = 5 
    
    # --- Part 1: ages 0-44, columns A-U. ---
    # A is region; B-U contain values.
    res_p1 = df.iloc[data_start_idx:, 0:21].copy()
    
    cols_p1 = ['地区']
    for g in age_groups_p1:
        cols_p1.extend([f"{g}_男", f"{g}_女"])
    
    res_p1.columns = cols_p1
    
    # --- Part 2: ages 45+, starting at column Y. ---
    # The 45+ block has 9 age groups times 2 gender columns.
    res_p2 = df.iloc[data_start_idx:, 24:24+18].copy()
    
    cols_p2 = []
    for g in age_groups_p2:
        cols_p2.extend([f"{g}_男", f"{g}_女"])
    
    res_p2.columns = cols_p2
    
    # --- Merge the two parts. ---
    df_combined = pd.concat([res_p1.reset_index(drop=True), res_p2.reset_index(drop=True)], axis=1)
    
    # --- Clean data. ---
    # 1. Remove whitespace from region names.
    df_combined['地区'] = df_combined['地区'].astype(str).str.replace(r'\s+', '', regex=True)
    
    # 2. Convert value columns to numeric.
    data_cols = [c for c in df_combined.columns if c != '地区']
    for col in data_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
    # 3. Keep prefecture-level cities only.
    df_city = df_combined[
        df_combined['地区'].str.endswith('市') & 
        ~df_combined['地区'].str.contains('合计|省|自治区|特别行政区|市辖区')
    ].copy()
    
    return df_city

# Run processing.
try:
    final_df_2020 = process_2020_census(file_path)
    
    # 4. Calculate the 2020 population aged 60+.
    older_cols = [c for c in final_df_2020.columns if any(x in c for x in ['60-64', '65-69', '70-74', '75-79', '80-84', '85岁'])]
    final_df_2020['2020_60Plus_Total'] = final_df_2020[older_cols].sum(axis=1)

    # Save results.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df_2020.to_excel(output_path, index=False)
    print("-" * 30)
    print("2020 census processing complete.")
    print(f"Prefecture-level city count: {len(final_df_2020)}")
    print(f"Saved to: {output_path}")

except Exception as e:
    import traceback
    print("Run failed:")
    traceback.print_exc()
