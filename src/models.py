"""
Phase 5 — Predictive Modeling & Empirical Evaluation Module
Climate-Smart Agriculture: Bi-Seasonal Paddy Yield Forecasting

Implements and benchmarks:
  1. Baseline Ordinary Least Squares (Linear Regression)
  2. Random Forest Regressor (Tuned with TimeSeriesSplit CV)
  3. LightGBM Regressor (Tuned with TimeSeriesSplit CV)
  4. Deep Multi-Layer Perceptron Neural Network (MLP / DNN)

Split Strategy:
  - Strict Chronological Train/Test Split (85% Train: 1950-2012, 15% Test: 2013-2024)
  - TimeSeriesSplit (5 Folds) cross-validation on the training partition

Outputs:
  - Serialized best model in models/best_model.pkl
  - Serialized all models dictionary in models/all_models.pkl
  - Benchmark comparison metrics table (RMSE, MAE, R2, Generalization Gap)
  - Diagnostic charts in reports/figures/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import Dict, Tuple, List, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prep_features(data_path: Path = None, test_ratio: float = 0.15) -> Dict[str, Any]:
    """
    Prepares feature matrices (X_train, X_test, y_train, y_test) using strict chronological ordering.
    """
    if data_path is None:
        data_path = DATA_PATH
        
    df = pd.read_csv(data_path)
    
    # Define model features (climate, macro, season, lags, interactions)
    feature_cols = [
        'season_binary',
        'rainfall_mm_scaled',
        'temperature_c_scaled',
        'gdp_billion_usd_scaled',
        'inflation_pct_scaled',
        'sown_000_acres_scaled',
        'harvested_000_acres_scaled',
        'yield_lag1',
        'yield_lag2',
        'rainfall_roll3',
        'temp_roll3',
        'rain_x_temp',
        'inf_x_gdp',
        'drought_flag',
        'harvest_efficiency'
    ]
    
    target_col = 'target_yield_production'
    
    # Chronological Split
    n_samples = len(df)
    n_train = int(n_samples * (1 - test_ratio))
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    
    return {
        'df': df,
        'train_df': train_df,
        'test_df': test_df,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def train_and_evaluate_all_models(save_best: bool = True) -> Dict[str, Any]:
    """
    Trains and tunes all 4 model architectures, evaluates on test set, and produces comparative visual diagnostics.
    """
    data = load_and_prep_features()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    feature_cols = data['feature_cols']
    test_df = data['test_df']
    
    print("=" * 80)
    print("PHASE 5: MACHINE LEARNING MODEL BENCHMARKING & EVALUATION")
    print(f"Training Instances: {len(X_train)} (1950 - {data['train_df']['year'].max()})")
    print(f"Test Instances (Holdout): {len(X_test)} ({data['test_df']['year'].min()} - {data['test_df']['year'].max()})")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print("=" * 80)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {}
    predictions = {}
    metrics = []
    
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Baseline Ordinary Least Squares (Linear Regression)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Training Baseline Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    
    # ──────────────────────────────────────────────────────────────────────────
    # 2. Random Forest Regressor (Hyperparameter Tuned with TimeSeriesSplit)
    # ──────────────────────────────────────────────────────────────────────────
    print("[2/4] Tuning Random Forest Regressor via TimeSeriesSplit CV...")
    rf_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8, 12, None],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    rf_cv = GridSearchCV(RandomForestRegressor(), rf_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    models['Random Forest'] = best_rf
    print(f"    - Best RF Params: {rf_cv.best_params_}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 3. LightGBM Regressor (Hyperparameter Tuned with TimeSeriesSplit)
    # ──────────────────────────────────────────────────────────────────────────
    print("[3/4] Tuning LightGBM Regressor via TimeSeriesSplit CV...")
    lgb_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'num_leaves': [7, 15, 31],
        'learning_rate': [0.03, 0.05, 0.1],
        'random_state': [42],
        'verbose': [-1]
    }
    lgb_cv = GridSearchCV(lgb.LGBMRegressor(), lgb_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    lgb_cv.fit(X_train, y_train)
    best_lgb = lgb_cv.best_estimator_
    models['LightGBM'] = best_lgb
    print(f"    - Best LightGBM Params: {lgb_cv.best_params_}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # 4. Deep Multi-Layer Perceptron (MLP / DNN)
    # ──────────────────────────────────────────────────────────────────────────
    print("[4/4] Training Deep Neural Network (3 Hidden Layers)...")
    mlp_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64, 32), (64, 64, 32)],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate_init': [0.005, 0.01],
        'max_iter': [800],
        'random_state': [42]
    }
    mlp_cv = GridSearchCV(MLPRegressor(early_stopping=True), mlp_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    mlp_cv.fit(X_train, y_train)
    best_mlp = mlp_cv.best_estimator_
    models['Deep Neural Network (MLP)'] = best_mlp
    print(f"    - Best MLP Params: {mlp_cv.best_params_}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Performance Evaluation & Metric Compilation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE EVALUATION ON HOLDOUT TEST SET")
    print("=" * 80)
    
    for name, model in models.items():
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        predictions[name] = test_pred
        
        # Test Metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Train Metrics (for overfitting analysis)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        generalization_gap = test_rmse - train_rmse
        
        metrics.append({
            'Model': name,
            'Test RMSE (000 Mt)': test_rmse,
            'Test MAE (000 Mt)': test_mae,
            'Test R²': test_r2,
            'Train R²': train_r2,
            'Generalization Gap (RMSE)': generalization_gap
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values('Test RMSE (000 Mt)').reset_index(drop=True)
    print(metrics_df.to_string(index=False))
    
    best_model_name = metrics_df.iloc[0]['Model']
    best_model = models[best_model_name]
    print(f"\n🏆 Best Performing Architecture: {best_model_name} (Test R² = {metrics_df.iloc[0]['Test R²']:.4f})")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Save Best Model Artifacts
    # ──────────────────────────────────────────────────────────────────────────
    if save_best:
        best_model_path = MODELS_DIR / "best_model.pkl"
        all_models_path = MODELS_DIR / "all_models.pkl"
        
        joblib.dump({
            'model': best_model,
            'model_name': best_model_name,
            'feature_cols': feature_cols,
            'metrics': metrics_df.to_dict(orient='records')
        }, best_model_path)
        
        joblib.dump({
            'models': models,
            'feature_cols': feature_cols,
            'metrics_df': metrics_df
        }, all_models_path)
        
        print(f"Saved best model serialized artifact to: {best_model_path}")
        print(f"Saved all models to: {all_models_path}")
        
    # ──────────────────────────────────────────────────────────────────────────
    # Generate Visualizations (Actual vs Predicted & Feature Importances)
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Predicted vs Actual Time Series Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    time_indices = [f"{y} {s}" for y, s in zip(test_df['year'], test_df['season'])]
    
    ax.plot(time_indices, y_test.values, 'k-o', label='Actual Production (Holdout)', linewidth=2.5, markersize=6)
    
    colors = {'Linear Regression': '#7f8c8d', 'Random Forest': '#27ae60', 'LightGBM': '#2980b9', 'Deep Neural Network (MLP)': '#8e44ad'}
    for name, preds in predictions.items():
        ax.plot(time_indices, preds, linestyle='--', marker='s', label=f"{name} (R²={r2_score(y_test, preds):.2f})", color=colors.get(name, '#e67e22'))
        
    ax.set_title('Holdout Validation: Actual vs Model Predicted Paddy Production (2013-2024)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Harvest Season')
    ax.set_ylabel('Production (000 Metric Tons)')
    ax.set_xticklabels(time_indices, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_model_predictions_vs_actual.png')
    plt.close()
    
    # 2. Feature Importance for Tree Models
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # RF Importance
    rf_imp = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    rf_imp.plot(kind='barh', ax=axes[0], color='#27ae60', edgecolor='black')
    axes[0].set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Gini Importance')
    
    # LightGBM Importance
    lgb_imp = pd.Series(best_lgb.feature_importances_, index=feature_cols).sort_values(ascending=True)
    lgb_imp.plot(kind='barh', ax=axes[1], color='#2980b9', edgecolor='black')
    axes[1].set_title('LightGBM Feature Importance', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Split Importance Weight')
    
    plt.suptitle('Predictive Feature Drivers of Bi-Seasonal Paddy Yield in Sri Lanka', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '08_feature_importance_comparison.png')
    plt.close()
    
    print(f"Generated and saved diagnostic figures in: {FIGURES_DIR}")
    
    return {
        'models': models,
        'metrics_df': metrics_df,
        'predictions': predictions,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'feature_cols': feature_cols,
        'data': data
    }


if __name__ == "__main__":
    results = train_and_evaluate_all_models()
