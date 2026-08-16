"""
Unit Tests for Bi-Seasonal Paddy Yield Forecasting Pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
import joblib

from src.data_loader import load_kaggle_rice_climate
from src.preprocessing import winsorize_series_iqr
from src.features import engineer_features

def test_kaggle_data_loading():
    df = load_kaggle_rice_climate()
    assert not df.empty
    assert len(df) >= 140
    assert 'year' in df.columns
    assert 'season' in df.columns
    assert 'production_000_mt' in df.columns
    assert df['production_000_mt'].isnull().sum() == 0

def test_winsorization():
    s = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 100.0, -50.0])
    capped, n_capped = winsorize_series_iqr(s)
    assert n_capped == 2
    assert capped.max() < 100.0
    assert capped.min() > -50.0

def test_feature_engineering():
    df_feat = engineer_features()
    assert not df_feat.empty
    assert 'yield_lag1' in df_feat.columns
    assert 'yield_lag2' in df_feat.columns
    assert 'rainfall_roll3' in df_feat.columns
    assert 'rain_x_temp' in df_feat.columns
    assert 'drought_flag' in df_feat.columns
    assert 'harvest_efficiency' in df_feat.columns
    assert df_feat['yield_lag1'].isnull().sum() == 0

def test_saved_model_inference():
    best_pkg_path = Path("models/best_model.pkl")
    assert best_pkg_path.exists()
    
    pkg = joblib.load(best_pkg_path)
    model = pkg['model']
    feature_cols = pkg['feature_cols']
    
    df_feat = pd.read_csv("data/processed/model_ready_features.csv")
    X_sample = df_feat[feature_cols].iloc[:5]
    
    preds = model.predict(X_sample)
    assert len(preds) == 5
    assert not np.isnan(preds).any()
    assert (preds > 0).all()
