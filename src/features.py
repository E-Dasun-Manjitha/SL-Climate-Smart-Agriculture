"""
Phase 4 — Feature Engineering Module
Climate-Smart Agriculture: Bi-Seasonal Paddy Yield Forecasting

Engineers domain-specific predictive features:
  1. Autoregressive Lag Features:
     - yield_lag1: Previous season's production (t-1)
     - yield_lag2: Previous year's same-season production (t-2)
  2. Multi-Season Rolling Moving Averages:
     - rainfall_roll3: 3-season rolling average rainfall
     - temp_roll3: 3-season rolling average temperature
  3. Non-Linear Interaction Terms:
     - rain_x_temp: Climate interaction (Rainfall * Temperature)
     - inf_x_gdp: Macroeconomic stress interaction (Inflation * GDP Growth)
  4. Climate Anomaly Indicators:
     - drought_flag: Binary flag where seasonal rainfall < (mean_season - 1.0 * std_season)
  5. Extent Efficiency Ratio:
     - harvest_efficiency: Ratio of harvested area to sown area (harvested / sown)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def engineer_features(input_path: Path = None, output_path: Path = None) -> pd.DataFrame:
    """
    Constructs advanced temporal, interaction, and climate-anomaly features.
    """
    if input_path is None:
        input_path = PROCESSED_DIR / "model_ready.csv"
    if output_path is None:
        output_path = PROCESSED_DIR / "model_ready_features.csv"
        
    df = pd.read_csv(input_path)
    
    # Sort chronologically by year and season to ensure correct lag calculations
    # Maha comes first in a given agricultural cycle, followed by Yala
    season_order = {'Maha': 0, 'Yala': 1}
    df['season_num'] = df['season'].map(season_order)
    df = df.sort_values(['year', 'season_num']).reset_index(drop=True)
    
    # 1. Autoregressive Lags
    # yield_lag1 (t-1 season)
    df['yield_lag1'] = df['target_yield_production'].shift(1)
    # yield_lag2 (t-2 season: exactly 1 year prior)
    df['yield_lag2'] = df['target_yield_production'].shift(2)
    
    # 2. Rolling Climate Averages (3-season rolling window)
    df['rainfall_roll3'] = df['rainfall_mm'].rolling(window=3, min_periods=1).mean()
    df['temp_roll3'] = df['temperature_c'].rolling(window=3, min_periods=1).mean()
    
    # 3. Non-Linear Interactions (on raw & standardized terms)
    df['rain_x_temp'] = df['rainfall_mm_scaled'] * df['temperature_c_scaled']
    df['inf_x_gdp'] = df['inflation_pct_scaled'] * df['gdp_billion_usd_scaled']
    
    # 4. Drought / Climate Anomaly Indicator (Season-specific threshold)
    # Calculate seasonal rainfall baseline (Maha vs Yala)
    seasonal_stats = df.groupby('season')['rainfall_mm'].agg(['mean', 'std']).reset_index()
    drought_flags = []
    for _, row in df.iterrows():
        s = row['season']
        rain = row['rainfall_mm']
        stats = seasonal_stats[seasonal_stats['season'] == s].iloc[0]
        threshold = stats['mean'] - 1.0 * stats['std']
        drought_flags.append(1 if rain < threshold else 0)
    df['drought_flag'] = drought_flags
    
    # 5. Harvest Efficiency Index (Harvested / Sown)
    df['harvest_efficiency'] = (df['harvested_000_acres'] / df['sown_000_acres']).clip(lower=0.5, upper=1.0)
    
    # Handle initial lag NaNs via backfill/median so no rows are lost
    df['yield_lag1'] = df['yield_lag1'].bfill()
    df['yield_lag2'] = df['yield_lag2'].bfill()
    
    # Drop temporary sorting helper
    df = df.drop(columns=['season_num'])
    
    # Save enriched dataset
    df.to_csv(output_path, index=False)
    print(f"[Features] Successfully engineered features. Shape: {df.shape}")
    print(f"[Features] Saved enriched dataset to: {output_path}")
    
    return df


if __name__ == "__main__":
    df_feat = engineer_features()
    print("\nFeature Summary:")
    print(df_feat[['year', 'season', 'yield_lag1', 'yield_lag2', 'rainfall_roll3', 'rain_x_temp', 'drought_flag', 'harvest_efficiency']].head(6).to_string())
