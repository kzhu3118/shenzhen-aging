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
    "北京", "上海", "广州", "东莞", 
    "长春",  "成都", "深圳"
]

CITY_LABELS = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "东莞": "Dongguan",
    "长春": "Changchun",
    "成都": "Chengdu",
    "深圳": "Shenzhen",
    "全国平均": "National average",
}

# Fixed city colors shared by observed and projected lines.
CITY_COLORS = {
    "北京": "#119131",   # dark blue
    "上海": "#2171b5",   # blue
    "广州": "#6baed6",   # light blue
    "东莞": "#fb6a4a",   # light red
    "长春": "#7f7f7f",   # gray
    "成都": "#e3a814",   # light gray
    "深圳": "#cb181d",   # dark red
    "全国平均": "#111111",
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

    target_df = df[df["city_short"].isin(TARGET_CITIES)].copy()
    if target_df.empty:
        raise ValueError("No target cities found in the input table.")

    return target_df, df


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


def calculate_aging_rate_change(target_df, all_df):
    """
    Calculate relative changes in aging rates from 2020 to 2035.
    The national average is computed as sum(pop_60_plus) / sum(total_pop).
    """
    city_rates = (
        target_df[target_df["year"].isin([2020, 2035])]
        .pivot_table(index="city_short", columns="year", values="aging_rate_60_plus", aggfunc="first")
        .reset_index()
    )
    city_rates = city_rates.dropna(subset=[2020, 2035]).copy()
    city_rates = city_rates[city_rates[2020] > 0].copy()
    city_rates["change_multiple"] = city_rates[2035] / city_rates[2020]
    city_rates["label"] = city_rates["city_short"].map(CITY_LABELS)
    city_rates["color"] = city_rates["city_short"].map(CITY_COLORS)

    national_rates = (
        all_df[all_df["year"].isin([2020, 2035])]
        .groupby("year")[["pop_60_plus", "total_pop"]]
        .sum()
    )
    national_rates["aging_rate_60_plus"] = national_rates["pop_60_plus"] / national_rates["total_pop"]

    if {2020, 2035}.issubset(national_rates.index) and national_rates.loc[2020, "aging_rate_60_plus"] > 0:
        national_change = (
            national_rates.loc[2035, "aging_rate_60_plus"]
            / national_rates.loc[2020, "aging_rate_60_plus"]
        )
        national_row = pd.DataFrame(
            {
                "city_short": ["全国平均"],
                2020: [national_rates.loc[2020, "aging_rate_60_plus"]],
                2035: [national_rates.loc[2035, "aging_rate_60_plus"]],
                "change_multiple": [national_change],
                "label": [CITY_LABELS["全国平均"]],
                "color": [CITY_COLORS["全国平均"]],
            }
        )
        city_rates = pd.concat([city_rates, national_row], ignore_index=True)

    city_rates = city_rates.sort_values("change_multiple", ascending=False).reset_index(drop=True)
    return city_rates


def calculate_national_aging_rates(all_df):
    national = (
        all_df.groupby("year")[["pop_60_plus", "total_pop"]]
        .sum()
        .sort_index()
    )
    national["aging_rate_60_plus"] = national["pop_60_plus"] / national["total_pop"]

    national_obs = national.reindex([2000, 2005, 2010, 2015, 2020]).copy()
    national_obs["aging_rate_60_plus"] = national_obs["aging_rate_60_plus"].interpolate(method="index")
    national_obs = national_obs.reset_index().rename(columns={"index": "year"})

    national_proj = national.loc[national.index.isin([2020, 2025, 2030, 2035])].copy()
    national_proj = national_proj.reset_index()
    return national_obs, national_proj


def plot_aging_rate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df, all_df = load_data()
    change_df = calculate_aging_rate_change(df, all_df)
    national_obs, national_proj = calculate_national_aging_rates(all_df)

    fig, (ax, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(13, 13),
        gridspec_kw={"height_ratios": [2.1, 1.2]},
    )

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

    national_color = CITY_COLORS["全国平均"]
    ax.plot(
        national_obs["year"],
        national_obs["aging_rate_60_plus"] * 100,
        linestyle="-",
        linewidth=4.0,
        marker="o",
        markersize=7.2,
        color=national_color,
        label=CITY_LABELS["全国平均"],
        zorder=5,
    )
    ax.plot(
        national_proj["year"],
        national_proj["aging_rate_60_plus"] * 100,
        linestyle="--",
        linewidth=3.6,
        marker="o",
        markersize=6.5,
        color=national_color,
        zorder=5,
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

    bars = ax_bar.bar(
        change_df["label"],
        change_df["change_multiple"],
        color=change_df["color"],
        alpha=0.9,
    )

    for bar in bars:
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}x",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax_bar.set_title("Relative Change in Aging Rate, 2020-2035", fontsize=18, pad=14)
    ax_bar.set_xlabel("City", fontsize=16)
    ax_bar.set_ylabel("Aging rate in 2035 / 2020", fontsize=16)
    ax_bar.tick_params(axis="x", labelsize=13, rotation=25)
    ax_bar.tick_params(axis="y", labelsize=13)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bar.set_axisbelow(True)

    upper = change_df["change_multiple"].max() * 1.12
    ax_bar.set_ylim(0, upper)

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
