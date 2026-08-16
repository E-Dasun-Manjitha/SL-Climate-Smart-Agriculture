"""
Phase 3 — Data Preprocessing Pipeline
Climate-Smart Agriculture: Bi-Seasonal Paddy Yield Forecasting

Handles:
  - Macro series evaluation and selection
  - Deduplication and data integrity checks
  - Missing value imputation (distribution-aware)
  - Noise & Outlier treatment via IQR-based winsorization (capping)
  - Seasonal categorical binarization (S_binary: Maha=1, Yala=0)
  - Chronological train/test aware standardization (StandardScaler)
  - Export of model-ready dataset to data/processed/model_ready.csv
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_primary, load_kaggle_rice_climate

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def winsorize_series_iqr(series: pd.Series, factor: float = 1.5) -> Tuple[pd.Series, int]:
    """
    Winsorize (cap) outliers in a numeric Series using IQR boundaries.
    Preserves sample size without dropping rows.
    
    Returns:
        (capped_series, count_of_capped_values)
    """
    s = series.copy()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    outlier_mask = (s < lower_bound) | (s > upper_bound)
    n_capped = outlier_mask.sum()
    
    s = s.clip(lower=lower_bound, upper=upper_bound)
    return s, int(n_capped)


def run_preprocessing_pipeline(
    train_ratio: float = 0.85,
    save_outputs: bool = True
) -> Dict[str, object]:
    """
    Executes the complete preprocessing pipeline.
    
    Args:
        train_ratio: Chronological train split fraction (default 0.85, last 15% holdout).
        save_outputs: Whether to save model_ready.csv and scaler.pkl to disk.
        
    Returns:
        dict containing cleaned DataFrame, train/test subsets, scalers, and transformation metadata.
    """
    print("=" * 70)
    print("EXECUTING PHASE 3 PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # 1. Load Primary Dataset
    df_raw = load_kaggle_rice_climate()
    initial_rows = len(df_raw)
    print(f"\n[1] Initial record count: {initial_rows} rows (1950 - {df_raw['year'].max()})")
    
    # 2. Cleaning: Deduplication
    df_clean = df_raw.drop_duplicates(subset=['year', 'season']).sort_values(
        ['year', 'season'], ascending=[True, True]
    ).reset_index(drop=True)
    dedup_rows = len(df_clean)
    print(f"[2] Deduplicated record count: {dedup_rows} rows (duplicates removed: {initial_rows - dedup_rows})")
    
    # 3. Handling Missing Values
    # In our dataset missing count is 0, but pipeline provides distribution-aware imputation:
    # Median for skewed variables (rainfall, inflation), Mean for normal variables (temperature, GDP)
    imputation_map = {
        'rainfall_mm': 'median',
        'inflation_pct': 'median',
        'temperature_c': 'mean',
        'gdp_billion_usd': 'median',
        'sown_000_acres': 'mean',
        'harvested_000_acres': 'mean',
        'production_000_mt': 'mean'
    }
    
    for col, method in imputation_map.items():
        if col in df_clean.columns and df_clean[col].isnull().any():
            if method == 'median':
                val = df_clean[col].median()
            else:
                val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(val)
            print(f"    - Imputed {col} using {method} ({val:.2f})")
            
    # 4. Outlier Treatment (IQR-based Winsorization)
    continuous_features = [
        'rainfall_mm', 'temperature_c', 'gdp_billion_usd', 'inflation_pct',
        'sown_000_acres', 'harvested_000_acres'
    ]
    
    capping_summary = {}
    for col in continuous_features:
        df_clean[col], n_capped = winsorize_series_iqr(df_clean[col])
        capping_summary[col] = n_capped
    print(f"[3] Outlier Winsorization Summary (capped at 1.5*IQR): {capping_summary}")
    
    # 5. Categorical Transformation (Seasonal Binarization)
    # Maha = 1 (Main monsoon season, major cultivation), Yala = 0 (Inter-monsoon season)
    df_clean['season_binary'] = df_clean['season'].apply(
        lambda s: 1 if str(s).strip().title() == 'Maha' else 0
    )
    print(f"[4] Season Binarization: Maha=1 ({sum(df_clean['season_binary']==1)} rows), Yala=0 ({sum(df_clean['season_binary']==0)} rows)")
    
    # 6. Chronological Train-Test Split for Scaling (No Data Leakage)
    n_train = int(len(df_clean) * train_ratio)
    train_years = df_clean.iloc[:n_train]['year']
    split_year = train_years.max()
    
    train_mask = df_clean.index < n_train
    test_mask = ~train_mask
    
    print(f"[5] Chronological Split: Train ({train_mask.sum()} rows, 1950-{split_year}), Test holdout ({test_mask.sum()} rows, {split_year+1}-{df_clean['year'].max()})")
    
    # Standardize continuous variables (fit ONLY on train set)
    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, continuous_features])
    
    # Create scaled columns
    scaled_cols = [f"{col}_scaled" for col in continuous_features]
    df_clean[scaled_cols] = scaler.transform(df_clean[continuous_features])
    
    # Rename target variable explicitly
    df_clean['target_yield_production'] = df_clean['production_000_mt']
    
    # 7. Save outputs
    if save_outputs:
        output_csv = PROCESSED_DIR / "model_ready.csv"
        df_clean.to_csv(output_csv, index=False)
        print(f"[6] Saved model-ready dataset to: {output_csv}")
        
        scaler_pkl = MODELS_DIR / "scaler.pkl"
        joblib.dump({
            'scaler': scaler,
            'features': continuous_features,
            'scaled_features': scaled_cols,
            'train_split_year': split_year
        }, scaler_pkl)
        print(f"[7] Saved fitted StandardScaler to: {scaler_pkl}")
        
    print("=" * 70)
    print("PHASE 3 PREPROCESSING COMPLETED SUCCESSFULLY")
    print(f"Final Model-Ready Shape: {df_clean.shape}")
    print("=" * 70)
    
    return {
        'df': df_clean,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'scaler': scaler,
        'continuous_features': continuous_features,
        'scaled_features': scaled_cols
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print(result['df'][['year', 'season', 'season_binary', 'rainfall_mm_scaled', 'temperature_c_scaled', 'target_yield_production']].head(8).to_string())
