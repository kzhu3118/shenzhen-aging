"""
Population-Weighted Accessibility Analysis
Purpose:
1. Compare population-weighted travel times with grid-unweighted means.
2. Check the 15-minute population coverage metric.
3. Produce evidence for the reviewer response.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR
FIGURE_DIR = PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load and merge data.
facility_distance = pd.read_csv(PROCESSED_DIR / 'facility_distance_time.csv')
pop_corrected = pd.read_csv(PROCESSED_DIR / 'pop_corrected.csv')

df = facility_distance.merge(pop_corrected, on='Id', how='inner')
print(f"Merged grid count: {len(df)}")

categories = [
    'restaurant', 'shopping', 'healthcare', 'tourist',
    'culture', 'sports', 'transportation', 'life', 'government'
]

# Handle avg_distance separately as the composite indicator.
THRESHOLD = 15  # 15-minute living-circle threshold.

# 2. Core functions.

def population_weighted_mean(time_series, pop_series):
    """Calculate a population-weighted mean after filtering missing and zero-population grids."""
    mask = time_series.notna() & pop_series.notna() & (pop_series > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(time_series[mask], weights=pop_series[mask])

def simple_mean(time_series):
    """Calculate the grid-unweighted mean."""
    return time_series.dropna().mean()

def coverage_rate(time_series, pop_series, threshold=THRESHOLD):
    """
    Calculate 15-minute population coverage as covered population divided by
    total population. This differs from the population-weighted mean.
    """
    mask_covered = (time_series <= threshold) & pop_series.notna()
    mask_total = pop_series.notna()
    covered_pop = pop_series[mask_covered].sum()
    total_pop = pop_series[mask_total].sum()
    if total_pop == 0:
        return np.nan
    return covered_pop / total_pop

# 3. Compare each facility type.

records = []

for cat in categories:
    elder_col = f'elder_time_{cat}'
    labor_col = f'labor_time_{cat}'

    # Older-adult scenario.
    elder_simple   = simple_mean(df[elder_col])
    elder_pw       = population_weighted_mean(df[elder_col], df['old_2020_sum'])
    elder_diff     = elder_pw - elder_simple
    elder_diff_pct = elder_diff / elder_simple * 100
    elder_cov      = coverage_rate(df[elder_col], df['old_2020_sum'])

    # Working-age scenario.
    labor_simple   = simple_mean(df[labor_col])
    labor_pw       = population_weighted_mean(df[labor_col], df['labor_2020_sum'])
    labor_diff     = labor_pw - labor_simple
    labor_diff_pct = labor_diff / labor_simple * 100
    labor_cov      = coverage_rate(df[labor_col], df['labor_2020_sum'])

    records.append({
        'Category'              : cat,
        # Older adults.
        'Elder_Simple_Mean(min)': round(elder_simple, 2),
        'Elder_PW_Mean(min)'    : round(elder_pw, 2),
        'Elder_Diff(min)'       : round(elder_diff, 2),
        'Elder_Diff(%)'         : round(elder_diff_pct, 1),
        'Elder_15min_Coverage'  : round(elder_cov, 4),
        # Working-age population.
        'Labor_Simple_Mean(min)': round(labor_simple, 2),
        'Labor_PW_Mean(min)'    : round(labor_pw, 2),
        'Labor_Diff(min)'       : round(labor_diff, 2),
        'Labor_Diff(%)'         : round(labor_diff_pct, 1),
        'Labor_15min_Coverage'  : round(labor_cov, 4),
    })

result_df = pd.DataFrame(records).set_index('Category')
print("\n======= Comparison Results =======")
print(result_df.to_string())
result_df.to_csv(str(OUTPUT_DIR / 'weighted_accessibility_comparison.csv'))

# 4. Weighted comparison for the composite avg_distance indicator.

for scenario, pop_col in [('elder', 'old_2020_sum'), ('labor', 'labor_2020_sum')]:
    col = f'{scenario}_time_avg_distance'
    if col in df.columns:
        s_mean = simple_mean(df[col])
        pw_mean = population_weighted_mean(df[col], df[pop_col])
        cov = coverage_rate(df[col], df[pop_col])
        print(f"\n[Composite indicator - {scenario}]")
        print(f"  Grid-unweighted mean: {s_mean:.2f} min")
        print(f"  Population-weighted mean: {pw_mean:.2f} min")
        print(f"  Difference: {pw_mean - s_mean:.2f} min ({(pw_mean-s_mean)/s_mean*100:.1f}%)")
        print(f"  15-minute coverage: {cov:.4f} ({cov*100:.2f}%)")

# 5. Flag notable differences.

print("\n======= Difference Check =======")
elder_diffs = result_df['Elder_Diff(%)'].abs()
labor_diffs = result_df['Labor_Diff(%)'].abs()

THRESHOLD_PCT = 5  # Treat differences within 5% as small.

elder_sig = elder_diffs[elder_diffs > THRESHOLD_PCT]
labor_sig = labor_diffs[labor_diffs > THRESHOLD_PCT]

if len(elder_sig) == 0 and len(labor_sig) == 0:
    print("All population-weighted means differ from grid means by less than 5%.")
    print("Coverage is already population-weighted, and unweighted results appear robust.")
else:
    print(f"Older-adult categories with differences above 5%: {list(elder_sig.index)}")
    print(f"Working-age categories with differences above 5%: {list(labor_sig.index)}")
    print("Include the population-weighted results in a supplementary table.")

# 6. Plot grid-unweighted vs population-weighted means.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cat_labels = [c.capitalize() for c in categories]

for ax, (scenario, simple_col, pw_col, title) in zip(axes, [
    ('Elder',  'Elder_Simple_Mean(min)', 'Elder_PW_Mean(min)',  'Older Adults (2.7 km/h)'),
    ('Labor',  'Labor_Simple_Mean(min)', 'Labor_PW_Mean(min)',  'Non-elderly (5.0 km/h)'),
]):
    x = result_df[simple_col].values
    y = result_df[pw_col].values

    ax.scatter(x, y, s=80, color='steelblue', zorder=3)
    for i, label in enumerate(cat_labels):
        ax.annotate(label, (x[i], y[i]), textcoords='offset points',
                    xytext=(5, 3), fontsize=8)

    # 1:1 reference line.
    lim_min = min(x.min(), y.min()) * 0.95
    lim_max = max(x.max(), y.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'r--', lw=1.2, label='1:1 reference')

    ax.set_xlabel('Grid-unweighted Mean Travel Time (min)', fontsize=10)
    ax.set_ylabel('Population-weighted Mean Travel Time (min)', fontsize=10)
    ax.set_title(f'{title}', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(alpha=0.3)

plt.suptitle('Grid-unweighted vs Population-weighted Mean Travel Time\n'
             '(Points near the 1:1 line indicate robustness of unweighted results)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(str(FIGURE_DIR / 'weighted_vs_unweighted_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\nFigure saved to {FIGURE_DIR / 'weighted_vs_unweighted_comparison.png'}")

# 7. Export supplementary-table format.

si_table = pd.DataFrame({
    'Facility Category': cat_labels,
    'Older Adults — Unweighted Mean (min)': result_df['Elder_Simple_Mean(min)'].values,
    'Older Adults — Pop-weighted Mean (min)': result_df['Elder_PW_Mean(min)'].values,
    'Older Adults — Difference (%)': result_df['Elder_Diff(%)'].values,
    'Non-elderly — Unweighted Mean (min)': result_df['Labor_Simple_Mean(min)'].values,
    'Non-elderly — Pop-weighted Mean (min)': result_df['Labor_PW_Mean(min)'].values,
    'Non-elderly — Difference (%)': result_df['Labor_Diff(%)'].values,
})
si_table.to_csv(str(OUTPUT_DIR / 'SI_Table_weighted_comparison.csv'), index=False)
print(f"Supplementary table saved to {OUTPUT_DIR / 'SI_Table_weighted_comparison.csv'}")
