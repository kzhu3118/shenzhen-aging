import traceback
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "city_age_panel_2000_2035_calibrated.xlsx"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_PNG = OUTPUT_DIR / "selected_cities_aging_rate_2000_2035_final.png"
OUTPUT_PDF = OUTPUT_DIR / "selected_cities_aging_rate_2000_2035_final.pdf"

TARGET_CITIES = [
    "北京", "上海", "广州", "东莞", "厦门",
    "长春", "武汉", "成都", "深圳"
]

CITY_LABELS = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "东莞": "Dongguan",
    "厦门": "Xiamen",
    "长春": "Changchun",
    "武汉": "Wuhan",
    "成都": "Chengdu",
    "深圳": "Shenzhen",
}

# Fixed city colors shared by observed and projected lines.
CITY_COLORS = {
    "北京": "#1f77b4",   # blue
    "上海": "#ff7f0e",   # orange
    "广州": "#2ca02c",   # green
    "东莞": "#d62728",   # red
    "厦门": "#9467bd",   # purple
    "长春": "#8c564b",   # brown
    "武汉": "#e377c2",   # pink
    "成都": "#7f7f7f",   # gray
    "深圳": "#bcbd22",   # olive
}


def normalize_city_name(name):
    name = str(name).strip()
    if name.endswith("市"):
        name = name[:-1]
    return name


def configure_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)
    df["city_short"] = df["city"].apply(normalize_city_name)

    if "aging_rate_60_plus" not in df.columns:
        df["aging_rate_60_plus"] = df["pop_60_plus"] / df["total_pop"]

    df = df[df["city_short"].isin(TARGET_CITIES)].copy()
    if df.empty:
        raise ValueError("No target cities found in the input table.")

    return df


def interpolate_observed_city(city_df):
    """
    Interpolate observed 2000, 2010, and 2020 values to five-year steps:
    2000, 2005, 2010, 2015, and 2020.
    """
    obs = city_df[city_df["year"].isin([2000, 2010, 2020])].copy()
    obs = obs.sort_values("year")

    target_years = [2000, 2005, 2010, 2015, 2020]
    obs = obs.set_index("year").reindex(target_years)
    obs["city"] = obs["city"].ffill().bfill()
    obs["city_short"] = obs["city_short"].ffill().bfill()

    obs["aging_rate_60_plus"] = obs["aging_rate_60_plus"].interpolate(method="index")
    obs = obs.reset_index().rename(columns={"index": "year"})
    return obs


def plot_aging_rate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = load_data()

    fig, ax = plt.subplots(figsize=(13, 8))

    for city_cn in TARGET_CITIES:
        city_df = df[df["city_short"] == city_cn].copy()
        if city_df.empty:
            continue

        city_label = CITY_LABELS[city_cn]
        city_color = CITY_COLORS[city_cn]

        # Observed 2000-2020 values after interpolation: solid line.
        obs_df = interpolate_observed_city(city_df)
        ax.plot(
            obs_df["year"],
            obs_df["aging_rate_60_plus"] * 100,
            linestyle="-",
            linewidth=2.8,
            marker="o",
            markersize=6.5,
            color=city_color,
            label=city_label,
        )

        # Projected 2020-2035 values: dashed line with the same city color.
        proj_df = city_df[city_df["year"].isin([2020, 2025, 2030, 2035])].copy()
        proj_df = proj_df.sort_values("year")

        ax.plot(
            proj_df["year"],
            proj_df["aging_rate_60_plus"] * 100,
            linestyle="--",
            linewidth=2.3,
            marker="o",
            markersize=5.8,
            color=city_color,
            alpha=0.95,
        )

    # Observed/projected boundary.
    ax.axvline(x=2020, color="black", linestyle=":", linewidth=1.2, alpha=0.85)
    ax.text(2020.4, ax.get_ylim()[1] * 0.96, "Observed / Projected", fontsize=12, color="black")

    ax.set_title("Share of Population Aged 60+ in Selected Cities, 2000-2035", fontsize=22, pad=16)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Population Aged 60+ (%)", fontsize=18)

    xticks = [2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035]
    ax.set_xticks(xticks)
    ax.tick_params(axis="both", labelsize=15)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="City", ncol=3, frameon=False, fontsize=14, title_fontsize=15)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    # plt.close(fig)

    print("-" * 40)
    print("Done.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    try:
        plot_aging_rate()
    except Exception:
        print("Plotting failed:")
        traceback.print_exc()
