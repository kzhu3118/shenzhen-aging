import pandas as pd
import numpy as np
from pathlib import Path

# 1. Set paths.
BASE_DIR = Path(__file__).resolve().parents[1]
file_path = BASE_DIR / 'data' / 'raw' / 'pop_census' / '2010_pop_census.xlsx'
output_path = BASE_DIR / 'data' / 'processed' / 'pop_census' / 'merged_national_city_age_2010.xlsx'

# 2. Define age groups.
age_groups_1 = ['0岁', '1-4岁', '5-9岁', '10-14岁', '15-19岁', '20-24岁', '25-29岁', '30-34岁', '35-39岁', '40-44岁']
age_groups_2 = ['45-49岁', '50-54岁', '55-59岁', '60-64岁', '65-69岁', '70-74岁', '75-79岁', '80-84岁', '85岁及以上']

def process_sheet(df, age_list):
    """Extract region and gender columns for two-column age groups."""
    res = pd.DataFrame()
    
    # The first four rows are titles/headers; data starts at row 5.
    data_start_idx = 4 
    
    # Extract the region column.
    res['地区'] = df.iloc[data_start_idx:, 0].astype(str).str.replace(r'\s+', '', regex=True)
    
    # Extract age columns; each age group has male and female columns.
    start_col = 1 
    for i, group in enumerate(age_list):
        male_idx = start_col + i * 2
        female_idx = start_col + i * 2 + 1
        
        # Guard against sheets with fewer columns.
        if male_idx < df.shape[1]:
            res[f"{group}_男"] = pd.to_numeric(df.iloc[data_start_idx:, male_idx], errors='coerce')
        if female_idx < df.shape[1]:
            res[f"{group}_女"] = pd.to_numeric(df.iloc[data_start_idx:, female_idx], errors='coerce')
            
    return res

# 3. Read all sheets and merge them in pairs.
try:
    excel_obj = pd.ExcelFile(file_path)
    all_sheets = excel_obj.sheet_names
    print(f"Detected sheets: {all_sheets}")
except Exception as e:
    print(f"Failed to read file: {e}")
    exit()

final_data_list = []

# Step by two to pair 0-44 sheets with 45+ sheets.
for i in range(0, len(all_sheets), 2):
    # Avoid overflow when the workbook has an odd number of sheets.
    if i + 1 >= len(all_sheets):
        break
        
    sheet_a_name = all_sheets[i]
    sheet_b_name = all_sheets[i+1]
    
    print(f"Merging: {sheet_a_name} (ages 0-44) + {sheet_b_name} (ages 45+)")
    
    # Read without headers.
    df_a = pd.read_excel(file_path, sheet_name=sheet_a_name, header=None)
    df_b = pd.read_excel(file_path, sheet_name=sheet_b_name, header=None)
    
    data_a = process_sheet(df_a, age_groups_1)
    data_b = process_sheet(df_b, age_groups_2)
    
    # Merge horizontally on region.
    merged_pair = pd.merge(data_a, data_b, on='地区')
    final_data_list.append(merged_pair)

if not final_data_list:
    print("No mergeable data found; check the sheet order.")
else:
    # 4. Stack vertically.
    full_df = pd.concat(final_data_list, ignore_index=True)

    # 5. Filter prefecture-level cities.
    df_city = full_df[
        full_df['地区'].str.endswith('市') & 
        ~full_df['地区'].str.contains('市辖区|合计|省|自治区')
    ].copy()

    # 6. Save.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_city.to_excel(output_path, index=False)
    print("-" * 30)
    print("Processing complete.")
    print(f"Prefecture-level city count: {len(df_city)}")
    print(f"Saved to: {output_path}")
