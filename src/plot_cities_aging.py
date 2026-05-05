import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "pop_census" / "city_census_panel_2000_2010_2020.xlsx"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_PNG = OUTPUT_DIR / "selected_cities_aging_rate_60plus.png"
OUTPUT_PDF = OUTPUT_DIR / "selected_cities_aging_rate_60plus.pdf"

TARGET_CITIES = ["北京", "上海", "广州", "东莞", "厦门", "长春", "武汉", "成都", "深圳"]
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


def normalize_city_name(city_name):
    city_name = str(city_name).strip()
    if city_name.endswith("市"):
        city_name = city_name[:-1]
    return city_name


def configure_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_plot_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find panel file: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)
    df["city_short"] = df["city"].apply(normalize_city_name)

    plot_df = df[df["city_short"].isin(TARGET_CITIES)].copy()
    if plot_df.empty:
        raise ValueError("No target cities matched; check the city column.")

    if "aging_rate_60_plus" not in plot_df.columns:
        plot_df["aging_rate_60_plus"] = plot_df["pop_60_plus"] / plot_df["total_pop"]

    plot_df["aging_rate_60_plus_pct"] = plot_df["aging_rate_60_plus"] * 100
    plot_df = plot_df.sort_values(["city_short", "year"])
    return plot_df


def plot_aging_rate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_df = load_plot_data()

    fig, ax = plt.subplots(figsize=(13, 8))

    for city in TARGET_CITIES:
        city_df = plot_df[plot_df["city_short"] == city].sort_values("year")
        if city_df.empty:
            continue
        ax.plot(
            city_df["year"],
            city_df["aging_rate_60_plus_pct"],
            marker="o",
            linewidth=2.6,
            markersize=7,
            label=CITY_LABELS[city],
        )

    ax.set_title("Share of Population Aged 60+ in Selected Cities", fontsize=22, pad=16)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Population Aged 60+ (%)", fontsize=18)
    ax.set_xticks(sorted(plot_df["year"].dropna().unique()))
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="City", ncol=3, frameon=False, fontsize=14, title_fontsize=15)

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    # plt.close(fig)

    print("-" * 30)
    print("Aging-rate line chart generated.")
    print(f"City count: {plot_df['city_short'].nunique()}")
    print(f"PNG output: {OUTPUT_PNG}")
    print(f"PDF output: {OUTPUT_PDF}")


if __name__ == "__main__":
    try:
        plot_aging_rate()
    except Exception:
        print("Plotting failed:")
        traceback.print_exc()
