import traceback
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "city_age_panel_2000_2050_calibrated.xlsx"
CHINA_DATA_FILE = BASE_DIR / "data" / "raw" / "china_elderly_population.csv"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_PNG = OUTPUT_DIR / "selected_cities_aging_rate_2000_2050_final.png"
OUTPUT_PDF = OUTPUT_DIR / "selected_cities_aging_rate_2000_2050_final.pdf"
AGING_SOCIETY_THRESHOLD = 10
PLOT_YEARS = [2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]

TARGET_CITIES = ["北京", "上海", "广州", "南京", "厦门", "长春", "武汉", "成都", "深圳", "茂名","东莞"]
CITY_LABELS = {
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "南京": "Nanjing",
    "厦门": "Xiamen", "长春": "Changchun", "武汉": "Wuhan", "成都": "Chengdu", "深圳": "Shenzhen",
     "茂名": "Maoming", "东莞": "Dongguan"
}
CITY_COLORS = {
    "北京": "#1f77b4", "上海": "#ff7f0e", "广州": "#2ca02c", "南京": "#bcbd22",
    "厦门": "#9467bd", "长春": "#8c564b", "武汉": "#e377c2", "成都": "#7f7f7f", "深圳": "#d62728",
     "茂名": "#b8860b", "东莞": "#BC3EA7"
}



def normalize_city_name(name):
    name = str(name).strip()
    return name[:-1] if name.endswith("市") else name


def configure_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def interpolate_observed_city(city_df):
    obs = city_df[city_df["year"].isin([2000, 2010, 2020])].copy().sort_values("year")
    target_years = [2000, 2005, 2010, 2015, 2020]
    obs = obs.set_index("year").reindex(target_years)
    obs["city"] = obs["city"].ffill().bfill()
    obs["city_short"] = obs["city_short"].ffill().bfill()
    obs["aging_rate_60_plus"] = obs["aging_rate_60_plus"].interpolate(method="index")
    return obs.reset_index().rename(columns={"index": "year"})


def load_china_aging_rate():
    china_df = pd.read_csv(CHINA_DATA_FILE)
    china_df = china_df[china_df["year"].isin(PLOT_YEARS)].copy().sort_values("year")
    return china_df[["year", "China_Pct_60+"]]


def order_legend_items(handles, labels):
    legend_items = dict(zip(labels, handles))
    bottom_labels = ["Shenzhen", "China"]
    ordered_labels = [label for label in labels if label not in bottom_labels]

    if all(label in legend_items for label in bottom_labels):
        left_column_count = (len(labels) + 1) // 2
        ordered_labels.insert(left_column_count - 1, "Shenzhen")
        ordered_labels.append("China")

    return [legend_items[label] for label in ordered_labels], ordered_labels


def plot_aging_rate():
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_FILE)
    df["city_short"] = df["city"].apply(normalize_city_name)

    fig, ax = plt.subplots(figsize=(13, 8))

    for city_cn in TARGET_CITIES:
        city_df = df[df["city_short"] == city_cn].copy()
        if city_df.empty:
            continue

        city_label = CITY_LABELS[city_cn]
        city_color = CITY_COLORS[city_cn]

        obs_df = interpolate_observed_city(city_df)
        ax.plot(
            obs_df["year"], obs_df["aging_rate_60_plus"] * 100,
            linestyle="-", linewidth=2.8, marker="o", markersize=6.5,
            color=city_color, label=city_label
        )

        proj_df = city_df[city_df["year"].isin([2020, 2025, 2030, 2035, 2040, 2045, 2050])].copy().sort_values("year")
        ax.plot(
            proj_df["year"], proj_df["aging_rate_60_plus"] * 100,
            linestyle="--", linewidth=2.3, marker="o", markersize=5.8,
            color=city_color, alpha=0.95
        )

    china_df = load_china_aging_rate()
    china_obs_df = china_df[china_df["year"] <= 2020]
    china_proj_df = china_df[china_df["year"] >= 2020]
    ax.plot(
        china_obs_df["year"], china_obs_df["China_Pct_60+"],
        linestyle="-", linewidth=3.4, marker="s", markersize=6.8,
        color="black", label="China", zorder=4
    )
    ax.plot(
        china_proj_df["year"], china_proj_df["China_Pct_60+"],
        linestyle="--", linewidth=3.0, marker="s", markersize=6.2,
        color="black", alpha=0.95, zorder=4
    )

    ax.axhline(
        y=AGING_SOCIETY_THRESHOLD,
        color="#6e6e6e",
        linestyle="--",
        linewidth=3.0,
        alpha=0.9,
        zorder=0,
    )
    ax.text(
        2049.5,
        AGING_SOCIETY_THRESHOLD + 0.55,
        "Aging society threshold: 10%",
        fontsize=12,
        color="#4f4f4f",
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )

    ax.axvline(x=2020, color="black", linestyle=":", linewidth=1.2, alpha=0.85)
    ax.text(2020.5, ax.get_ylim()[1] * 0.96, "Census / Projected", fontsize=12, color="black")

    # ax.set_title("Share of Population Aged 60+ in Selected Cities, 2000-2050", fontsize=22, pad=16)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Older adults population percentage (%)", fontsize=18)
    ax.set_xticks(PLOT_YEARS)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = order_legend_items(handles, labels)
    ax.legend(handles, labels, title="City", ncol=2, frameon=False, fontsize=14, title_fontsize=15)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    # plt.close(fig)

    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    try:
        plot_aging_rate()
    except Exception:
        print("Plotting failed:")
        traceback.print_exc()
