"""
Phase 2 — Exploratory Data Analysis (EDA)

Generates all EDA visualizations and summary statistics for the 
bi-seasonal paddy yield forecasting pipeline.

Outputs:
  - All figures saved to reports/figures/
  - Summary printed to console
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Try importing seaborn — fallback gracefully if not yet installed
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("WARNING: seaborn not installed — using matplotlib only")

from pathlib import Path
from src.data_loader import load_primary

# ── Configuration ──
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
SEASON_COLORS = {'Yala': '#E07A3A', 'Maha': '#3A7EBF'}
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading datasets...")
datasets = load_primary()
kaggle = datasets['kaggle_rice_climate']
dcs_yield = datasets['dcs_paddy_yield_district']
dcs_extent = datasets['dcs_paddy_extent_district']
dcs_national = datasets['dcs_paddy_extent_national']

print(f"\nKaggle: {kaggle.shape}, DCS Yield: {dcs_yield.shape}, "
      f"DCS Extent: {dcs_extent.shape}, DCS National: {dcs_national.shape}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Univariate Analysis...")

continuous_cols = ['rainfall_mm', 'temperature_c', 'production_000_mt',
                   'sown_000_acres', 'harvested_000_acres',
                   'gdp_billion_usd', 'inflation_pct']
col_labels = {
    'rainfall_mm': 'Rainfall (mm)',
    'temperature_c': 'Temperature (C)',
    'production_000_mt': 'Production (000 Mt)',
    'sown_000_acres': 'Sown (000 Acres)',
    'harvested_000_acres': 'Harvested (000 Acres)',
    'gdp_billion_usd': 'GDP (Billion USD)',
    'inflation_pct': 'Inflation (%)',
}

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    ax = axes[i]
    for season, color in SEASON_COLORS.items():
        data = kaggle[kaggle['season'] == season][col].dropna()
        ax.hist(data, bins=20, alpha=0.6, color=color, label=season, edgecolor='white')
    ax.set_title(col_labels.get(col, col))
    ax.set_xlabel(col_labels.get(col, col))
    ax.set_ylabel('Frequency')
    ax.legend()

# Season record counts
ax = axes[7]
season_counts = kaggle['season'].value_counts()
bars = ax.bar(season_counts.index, season_counts.values,
              color=[SEASON_COLORS.get(s, '#888') for s in season_counts.index],
              edgecolor='white', linewidth=1.5)
ax.set_title('Record Count by Season')
ax.set_ylabel('Count')
for bar, val in zip(bars, season_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontweight='bold')

# Hide extra subplot
axes[8].set_visible(False)

fig.suptitle('Univariate Analysis — Kaggle Rice & Climate Dataset', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGURES_DIR / '01_univariate_histograms.png')
plt.close()
print("  Saved: 01_univariate_histograms.png")


# ══════════════════════════════════════════════════════════════════════════════
#  2. OUTLIER DETECTION (Box Plots)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/7] Outlier Detection...")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

outlier_report = {}

for i, col in enumerate(continuous_cols):
    ax = axes[i]
    data = kaggle[col].dropna()
    
    # IQR outlier detection
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((data < lower) | (data > upper)).sum()
    outlier_report[col] = n_outliers
    
    # Box plot by season
    if HAS_SEABORN:
        sns.boxplot(data=kaggle, x='season', y=col, ax=ax,
                    palette=SEASON_COLORS, width=0.5)
    else:
        yala_data = kaggle[kaggle['season'] == 'Yala'][col].dropna()
        maha_data = kaggle[kaggle['season'] == 'Maha'][col].dropna()
        bp = ax.boxplot([yala_data, maha_data], labels=['Yala', 'Maha'], patch_artist=True)
        bp['boxes'][0].set_facecolor(SEASON_COLORS['Yala'])
        bp['boxes'][1].set_facecolor(SEASON_COLORS['Maha'])
    
    ax.set_title(f'{col_labels.get(col, col)}\n({n_outliers} outliers)')
    ax.set_xlabel('')

axes[7].set_visible(False)

fig.suptitle('Outlier Detection — Box Plots with IQR Method', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGURES_DIR / '02_outlier_boxplots.png')
plt.close()

print("  Outlier counts (1.5*IQR):")
for col, n in outlier_report.items():
    print(f"    {col_labels.get(col, col)}: {n} outliers")
print("  Saved: 02_outlier_boxplots.png")


# ══════════════════════════════════════════════════════════════════════════════
#  3. MISSING DATA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("[3/7] Missing Data Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Kaggle — no missing, but show the bar chart as confirmation
ax = axes[0]
missing_kaggle = kaggle.isnull().sum()
colors = ['#2ecc71' if v == 0 else '#e74c3c' for v in missing_kaggle.values]
ax.barh(missing_kaggle.index, missing_kaggle.values, color=colors, edgecolor='white')
ax.set_title('Kaggle Dataset — Missing Values per Column')
ax.set_xlabel('Missing Count')
for i, v in enumerate(missing_kaggle.values):
    ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')

# DCS District Yield — has some missing
ax = axes[1]
missing_dcs = dcs_yield.isnull().sum()
colors = ['#2ecc71' if v == 0 else '#e74c3c' for v in missing_dcs.values]
ax.barh(missing_dcs.index, missing_dcs.values, color=colors, edgecolor='white')
ax.set_title('DCS Yield District — Missing Values per Column')
ax.set_xlabel('Missing Count')
for i, v in enumerate(missing_dcs.values):
    ax.text(v + 5, i, str(v), va='center', fontweight='bold')

fig.suptitle('Missing Data Summary', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(FIGURES_DIR / '03_missing_data.png')
plt.close()
print("  Saved: 03_missing_data.png")


# ══════════════════════════════════════════════════════════════════════════════
#  4. DUPLICATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("[4/7] Duplicate Detection...")

n_exact_dupes = kaggle.duplicated().sum()
n_key_dupes = kaggle.duplicated(subset=['year', 'season']).sum()
print(f"  Kaggle — Exact duplicate rows: {n_exact_dupes}")
print(f"  Kaggle — Duplicate (year, season) combos: {n_key_dupes}")

n_exact_dupes_dcs = dcs_national.duplicated().sum()
n_key_dupes_dcs = dcs_national.duplicated(subset=['year', 'season']).sum()
print(f"  DCS National — Exact duplicate rows: {n_exact_dupes_dcs}")
print(f"  DCS National — Duplicate (year, season) combos: {n_key_dupes_dcs}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. TIME TREND: Paddy Yield/Production Over Time
# ══════════════════════════════════════════════════════════════════════════════
print("[5/7] Time Trend Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Production over time by season (Kaggle)
ax = axes[0, 0]
for season, color in SEASON_COLORS.items():
    mask = kaggle['season'] == season
    ax.plot(kaggle.loc[mask, 'year'], kaggle.loc[mask, 'production_000_mt'],
            color=color, label=season, linewidth=1.5, marker='.', markersize=4)
ax.set_title('National Paddy Production Over Time (Kaggle)')
ax.set_xlabel('Year')
ax.set_ylabel('Production (000 Mt)')
ax.legend()
ax.grid(True, alpha=0.3)

# Rainfall over time by season
ax = axes[0, 1]
for season, color in SEASON_COLORS.items():
    mask = kaggle['season'] == season
    ax.plot(kaggle.loc[mask, 'year'], kaggle.loc[mask, 'rainfall_mm'],
            color=color, label=season, linewidth=1.5, marker='.', markersize=4)
ax.set_title('Seasonal Rainfall Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Rainfall (mm)')
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature over time
ax = axes[1, 0]
for season, color in SEASON_COLORS.items():
    mask = kaggle['season'] == season
    ax.plot(kaggle.loc[mask, 'year'], kaggle.loc[mask, 'temperature_c'],
            color=color, label=season, linewidth=1.5, marker='.', markersize=4)
ax.set_title('Seasonal Temperature Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (C)')
ax.legend()
ax.grid(True, alpha=0.3)

# GDP over time
ax = axes[1, 1]
ax.plot(kaggle['year'], kaggle['gdp_billion_usd'],
        color='#2c3e50', linewidth=1.5, marker='.', markersize=4)
ax.set_title('GDP Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('GDP (Billion USD)')
ax.grid(True, alpha=0.3)

fig.suptitle('Time Trend Analysis — Key Variables', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIGURES_DIR / '04_time_trends.png')
plt.close()
print("  Saved: 04_time_trends.png")


# ══════════════════════════════════════════════════════════════════════════════
#  6. CONSISTENCY CHECK: Kaggle vs DCS National
# ══════════════════════════════════════════════════════════════════════════════
print("[6/7] Consistency Check — Kaggle vs DCS National...")

# Merge on (year, season) where both have production data
kaggle_prod = kaggle[['year', 'season', 'production_000_mt']].rename(
    columns={'production_000_mt': 'kaggle_prod'})
dcs_prod = dcs_national[['year', 'season', 'production_000_mt']].rename(
    columns={'production_000_mt': 'dcs_prod'})

merged = pd.merge(kaggle_prod, dcs_prod, on=['year', 'season'], how='inner')
merged['diff_pct'] = ((merged['kaggle_prod'] - merged['dcs_prod']) / merged['dcs_prod'] * 100).abs()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter: Kaggle vs DCS production
ax = axes[0]
for season, color in SEASON_COLORS.items():
    mask = merged['season'] == season
    ax.scatter(merged.loc[mask, 'dcs_prod'], merged.loc[mask, 'kaggle_prod'],
               color=color, label=season, alpha=0.7, s=30)
max_val = max(merged['dcs_prod'].max(), merged['kaggle_prod'].max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect agreement')
ax.set_title('Production: Kaggle vs DCS National')
ax.set_xlabel('DCS National Production (000 Mt)')
ax.set_ylabel('Kaggle Production (000 Mt)')
ax.legend()
ax.grid(True, alpha=0.3)

# Percentage difference over time
ax = axes[1]
for season, color in SEASON_COLORS.items():
    mask = merged['season'] == season
    ax.plot(merged.loc[mask, 'year'], merged.loc[mask, 'diff_pct'],
            color=color, label=season, linewidth=1.5, marker='.', markersize=4)
ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
ax.set_title('Absolute % Difference Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('|Difference| (%)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Consistency Check — Kaggle vs DCS National Production',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(FIGURES_DIR / '05_consistency_check.png')
plt.close()

# Report disagreements
big_diffs = merged[merged['diff_pct'] > 5]
print(f"  Overlapping year-season records: {len(merged)}")
print(f"  Mean absolute % difference: {merged['diff_pct'].mean():.2f}%")
print(f"  Records with >5% disagreement: {len(big_diffs)}")
if len(big_diffs) > 0:
    print(f"  Years with notable disagreement:")
    for _, row in big_diffs.head(10).iterrows():
        print(f"    {int(row['year'])} {row['season']}: "
              f"Kaggle={row['kaggle_prod']:.0f}, DCS={row['dcs_prod']:.0f}, "
              f"diff={row['diff_pct']:.1f}%")
print("  Saved: 05_consistency_check.png")


# ══════════════════════════════════════════════════════════════════════════════
#  7. CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("[7/7] Correlation Matrix...")

corr_cols = ['rainfall_mm', 'temperature_c', 'gdp_billion_usd', 'inflation_pct',
             'sown_000_acres', 'harvested_000_acres', 'production_000_mt']
corr_labels = [col_labels.get(c, c) for c in corr_cols]

corr_matrix = kaggle[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
if HAS_SEABORN:
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=corr_labels, yticklabels=corr_labels,
                square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
else:
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_labels)))
    ax.set_yticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right')
    ax.set_yticklabels(corr_labels)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f'{corr_matrix.values[i,j]:.2f}',
                    ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_title('Feature Correlation Matrix (Kaggle Dataset)',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_correlation_matrix.png')
plt.close()
print("  Saved: 06_correlation_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EDA SUMMARY — KEY FINDINGS")
print("=" * 70)

print(f"""
1. DATASET OVERVIEW:
   - Kaggle dataset: {kaggle.shape[0]} records, {kaggle.shape[1]} columns (1950-2024)
   - Covers both Yala ({(kaggle['season']=='Yala').sum()} records) and Maha ({(kaggle['season']=='Maha').sum()} records) seasons
   - Zero missing values in Kaggle dataset
   - DCS district data has some missing values in early years (pre-1983)

2. DATA QUALITY:
   - Exact duplicates (Kaggle): {n_exact_dupes}
   - Duplicate (year, season) keys (Kaggle): {n_key_dupes}
   - Outlier counts (1.5*IQR method): {outlier_report}

3. CONSISTENCY (Kaggle vs DCS National):
   - {len(merged)} overlapping year-season records compared
   - Mean |difference|: {merged['diff_pct'].mean():.2f}%
   - Records with >5% disagreement: {len(big_diffs)}

4. KEY CORRELATIONS WITH PRODUCTION:
""")

prod_corr = kaggle[corr_cols].corr()['production_000_mt'].drop('production_000_mt').sort_values(ascending=False)
for feat, val in prod_corr.items():
    direction = "positive" if val > 0 else "negative"
    strength = "strong" if abs(val) > 0.5 else "moderate" if abs(val) > 0.3 else "weak"
    print(f"   {col_labels.get(feat, feat):30s}: r = {val:+.3f} ({strength} {direction})")

print(f"""
5. RECOMMENDATIONS FOR PREPROCESSING:
   - Use KAGGLE dataset as authoritative target (Y_seasonal) — clean, complete, long history
   - DCS National can serve as cross-validation reference
   - DCS District data useful if district-level modeling is pursued
   - Climate data appears complete in Kaggle — supplemental weather dataset likely NOT needed
   - GDP and Inflation already present in Kaggle — CBSL macro data useful for cross-validation
""")

print("\nAll figures saved to reports/figures/")
print("EDA complete!")
