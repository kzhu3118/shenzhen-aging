import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

# =========================
# 1. Set paths.
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
WORLDPOP_DIR = Path("/Volumes/拯救者PSSD/Aexercise/data/GEO_data/World_pop_age")
CITY_SHP = BASE_DIR / "data" / "raw" / "china_city_filtered_297.shp"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# True: use constrained_UNadj files.
# False: use constrained files.
USE_UNADJ = True

# =========================
# 2. Read city shapefile.
# =========================
china_city = gpd.read_file(CITY_SHP)

# Keep required fields only.
base_cols = ["省", "省代码", "市", "市代码", "市类型", "省类型", "geometry"]
china_city = china_city[base_cols].copy()

# Assign a stable city id for later merges.
china_city = china_city.reset_index(drop=True)
china_city["city_id"] = china_city.index

# =========================
# 3. Collect tif files.
# =========================
if USE_UNADJ:
    tif_files = sorted(WORLDPOP_DIR.glob("chn_*_2020_constrained_UNadj.tif"))
else:
    tif_files = sorted(WORLDPOP_DIR.glob("chn_*_2020_constrained.tif"))

# Skip macOS resource-fork files.
tif_files = [f for f in tif_files if not f.name.startswith("._")]

if not tif_files:
    raise FileNotFoundError("No matching tif files found; check the path or filenames.")

print(f"Found {len(tif_files)} valid raster files.")

# =========================
# 4. Filename parser.
# Example: chn_f_25_2020_constrained_UNadj.tif
# =========================
def parse_worldpop_filename(fname: str):
    m = re.match(r"chn_([fm])_(\d+)_2020_constrained(?:_UNadj)?\.tif$", fname)
    if not m:
        raise ValueError(f"Cannot parse filename: {fname}")
    sex_code, age_code = m.groups()
    sex = "female" if sex_code == "f" else "male"
    age_start = int(age_code)

    # WorldPop age-band labels.
    if age_start == 0:
        age_label = "0"
    elif age_start == 1:
        age_label = "1-4"
    elif age_start == 80:
        age_label = "80+"
    else:
        age_label = f"{age_start}-{age_start+4}"

    return sex, age_start, age_label

# =========================
# 5. Align projections.
# =========================
with rasterio.open(tif_files[0]) as src:
    raster_crs = src.crs
    nodata = src.nodata

print("Raster CRS:", raster_crs)
print("Raster nodata:", nodata)

if china_city.crs != raster_crs:
    china_city = china_city.to_crs(raster_crs)

# =========================
# 6. Run city-level zonal sums for each raster.
# =========================
long_results = []

for tif in tif_files:
    sex, age_start, age_label = parse_worldpop_filename(tif.name)
    print(f"Processing: {tif.name} -> sex={sex}, age={age_label}")

    zs = zonal_stats(
        vectors=china_city,
        raster=str(tif),
        stats=["sum"],
        nodata=nodata,
        geojson_out=False,
        all_touched=False,
    )

    temp = china_city[["city_id", "省", "省代码", "市", "市代码", "市类型", "省类型"]].copy()
    temp["sex"] = sex
    temp["age_start"] = age_start
    temp["age_group"] = age_label
    temp["population"] = [x["sum"] if x["sum"] is not None else 0 for x in zs]

    long_results.append(temp)

long_df = pd.concat(long_results, ignore_index=True)

# =========================
# 7. Build the wide table.
# Example columns: female_0, female_1-4, male_25-29.
# =========================
long_df["col_name"] = long_df["sex"] + "_" + long_df["age_group"]

wide_df = (
    long_df.pivot_table(
        index=["city_id", "省", "省代码", "市", "市代码", "市类型", "省类型"],
        columns="col_name",
        values="population",
        aggfunc="first",
        fill_value=0,
    )
    .reset_index()
)

wide_df.columns.name = None

# =========================
# 8. Sort wide-table columns by age.
# =========================
age_order = ["0", "1-4"] + [f"{i}-{i+4}" for i in range(5, 80, 5)] + ["80+"]
ordered_cols = []
for sex in ["female", "male"]:
    for age in age_order:
        c = f"{sex}_{age}"
        if c in wide_df.columns:
            ordered_cols.append(c)

meta_cols = ["city_id", "省", "省代码", "市", "市代码", "市类型", "省类型"]
wide_df = wide_df[meta_cols + ordered_cols]

# =========================
# 9. Export.
# =========================
long_out_csv = OUTPUT_DIR / "china_city_worldpop_age_2020_long.csv"
wide_out_csv = OUTPUT_DIR / "china_city_worldpop_age_2020_wide.csv"
long_out_xlsx = OUTPUT_DIR / "china_city_worldpop_age_2020_long.xlsx"
wide_out_xlsx = OUTPUT_DIR / "china_city_worldpop_age_2020_wide.xlsx"

long_df.to_csv(long_out_csv, index=False, encoding="utf-8-sig")
wide_df.to_csv(wide_out_csv, index=False, encoding="utf-8-sig")
long_df.to_excel(long_out_xlsx, index=False)
wide_df.to_excel(wide_out_xlsx, index=False)

print("Done.")
print("Long table:", long_out_csv)
print("Wide table:", wide_out_csv)
