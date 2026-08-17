# 🌾 Climate-Smart Agriculture: Bi-Seasonal Paddy Yield Forecasting System
### Using Hybrid Predictive Data Mining and Macroeconomic Regression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-brightgreen)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Ensemble%20ML-Random%20Forest%20%7C%20LightGBM-orange)](#-empirical-benchmark-results)
[![Academic Module](https://img.shields.io/badge/Module-IT41033%20NIA-purple)](#-academic-metadata)
[![Test Suite](https://img.shields.io/badge/pytest-4%20passed-success)](#-automated-testing)

---

## 📌 Executive Summary
Rice (*Oryza sativa*) cultivation is the economic and nutritional cornerstone of Sri Lanka, sustaining over 1.8 million rural farming families. However, national paddy production suffers from severe bi-seasonal volatility driven by monsoonal shifts (**Maha** vs **Yala** cycles) and compounding macroeconomic shocks (inflation, currency depreciation, and input supply disruptions).

This repository contains an **end-to-end, reproducible Machine Learning forecasting pipeline** built upon **75 years (1950–2024)** of historical agrometeorological and macroeconomic records ($N = 149$ seasons). The system benchmarks tree-based ensembles (**Random Forest**, **LightGBM**) against **Ordinary Least Squares (Linear Regression)** and **Deep Neural Networks (MLP)**, deployed alongside an interactive **Streamlit decision-support web application**.

---

## 🎯 Research Questions & Objectives

1. **RQ1 (Leakage-Free Multi-Modal Preprocessing):** *How can heterogeneous continuous metrics (precipitation, temperature, inflation, GDP) and qualitative seasonal labels be preprocessed without lookahead data leakage?*
   - **Solution:** Distribution-aware missing imputation (median/mean), Interquartile Range (IQR) winsorization, seasonal binary encoding ($S_{binary} \in \{0, 1\}$), and strict chronological train-test isolation for Z-Score scaling ($StandardScaler$).
2. **RQ2 (Ensemble Trees vs. Deep Learning on Tabular Series):** *Do advanced ensemble methods outperform traditional deep neural networks on multi-decade tabular agrometeorological time-series?*
   - **Empirical Resolution:** **Random Forest** ($R^2 = 0.5767$) and **LightGBM** ($R^2 = 0.5064$) substantially outperform the **Deep Neural Network** ($R^2 = 0.1790$) due to superior inductive bias on small tabular sample regimes ($N = 149$).

---

## 📊 System Architecture & Mathematical Formulation

The seasonal national production target ($Y_{seasonal}$, thousand metric tons) is formulated as:
$$Y_{seasonal} = f\left(C_{rain}, C_{temp}, E_{gdp}, E_{inf}, S_{binary}, X_{engineered}\right)$$

```
Raw Multi-Source Data (DCS, Kaggle, CBSL, Met Dept)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. INGESTION & CONSISTENCY VALIDATION (src/data_loader.py)  │
│    • Kaggle vs DCS National check (0.02% mean difference)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA PREPROCESSING & CLEANING (src/preprocessing.py)     │
│    • Missing value imputation (Median for skewed, Mean)     │
│    • IQR Winsorization (Outlier preservation: 1.5*IQR)      │
│    • Seasonal Binarization: Maha = 1, Yala = 0              │
│    • Chronological Partitioning: Train (85%) | Test (15%)   │
│    • StandardScaler fitted ONLY on Training Set (1950-2012) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DOMAIN FEATURE ENGINEERING (src/features.py)             │
│    • Autoregressive Lags: yield_lag1 (t-1), yield_lag2 (t-2)│
│    • Rolling Climate Means: rainfall_roll3, temp_roll3      │
│    • Non-linear Interactions: rain_x_temp, inf_x_gdp        │
│    • Climate Anomaly: drought_flag (>1.0 std below mean)    │
│    • Land Efficiency: harvest_efficiency (harvested / sown) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PREDICTIVE MODELING & BENCHMARKING (src/models.py)       │
│    • 5-Fold TimeSeriesSplit Cross-Validation on Train Split │
│    • Hyperparameter Tuning (Random Forest, LightGBM, MLP)   │
│    • Holdout Test Evaluation (2013–2024 unseen data)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. INTERACTIVE STREAMLIT DASHBOARD (dashboard/app.py)       │
│    • Real-time Prediction KPI Card & Historical Delta %     │
│    • 4 Diagnostic Visual Tabs & Scenario Simulator          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏆 Empirical Benchmark Results

Evaluated strictly on the **unseen Holdout Test Partition (2013–2024, 23 cultivation seasons)**:

| Rank | Model Architecture | Test RMSE (000 Mt) | Test MAE (000 Mt) | Test $R^2$ | Train $R^2$ | Generalization Gap (RMSE) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest Regressor** | **419.43** | 376.02 | **0.5767** | 0.9892 | 356.16 |
| 🥈 | **LightGBM Regressor** | 452.94 | 365.35 | **0.5064** | 0.9873 | 384.36 |
| 🥉 | **Baseline Linear Regression** | 496.27 | **330.76** | 0.4075 | 0.9793 | 408.61 |
| 4 | **Deep Neural Network (MLP)** | 584.15 | 505.49 | 0.1790 | 0.8901 | 382.37 |

---

## 📁 Repository Directory Structure

```
SL-Climate-Smart-Agriculture/
├── data/
│   ├── raw/                  # Raw input datasets (untracked in git)
│   ├── interim/              # Temporary transformation cache
│   └── processed/            # Final model-ready datasets
│       ├── model_ready.csv          # Preprocessed base dataset (149 rows, 17 cols)
│       └── model_ready_features.csv # Enriched dataset (149 rows, 25 cols)
├── notebooks/                # Sequential, fully executable Jupyter Notebooks
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & Consistency Checks
│   ├── 02_preprocessing.ipynb# Data Cleaning, IQR Winsorization & Scaling
│   ├── 03_feature_engineering.ipynb # Lags, Rolling Means & Interaction Signals
│   ├── 04_modeling.ipynb     # TimeSeriesSplit CV Model Training
│   └── 05_evaluation.ipynb   # Benchmark Comparison & Generalization Analysis
├── src/                      # Modular production Python package
│   ├── __init__.py
│   ├── data_loader.py        # Ingestion for Excel & CSV data streams
│   ├── eda.py                # EDA visualization generator
│   ├── preprocessing.py      # End-to-end preprocessing pipeline
│   ├── features.py           # Feature engineering module
│   └── models.py             # Model training & benchmarking engine
├── dashboard/
│   └── app.py                # Multi-page interactive Streamlit Dashboard
├── reports/
│   ├── figures/              # 8 High-Resolution Publication Charts (PNG)
│   │   ├── 01_univariate_histograms.png
│   │   ├── 02_outlier_boxplots.png
│   │   ├── 03_missing_data.png
│   │   ├── 04_time_trends.png
│   │   ├── 05_consistency_check.png
│   │   ├── 06_correlation_matrix.png
│   │   ├── 07_model_predictions_vs_actual.png
│   │   └── 08_feature_importance_comparison.png
│   ├── final_report.md       # Comprehensive academic final report
│   ├── final_report.pdf      # Exported publication-ready PDF report
│   ├── individual_contribution_template.md # Group contribution matrix
│   └── github_collaboration_guide.md       # Team Git branch instructions
├── models/
│   ├── best_model.pkl        # Serialized best model (Random Forest)
│   ├── all_models.pkl        # All 4 trained model architectures
│   └── scaler.pkl            # Fitted StandardScaler
├── tests/
│   └── test_pipeline.py      # Automated pytest test suite
├── requirements.txt          # Frozen dependency specifications
├── .gitignore                # Optimized Git ignore configuration
└── README.md                 # Complete project documentation
```

---

## ⚡ Quick Start & Setup Instructions

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/E-Dasun-Manjitha/SL-Climate-Smart-Agriculture.git
cd SL-Climate-Smart-Agriculture

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\activate

# Activate environment (Linux / macOS)
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```bash
python -m pytest tests/
```

---

## 🔄 End-to-End Pipeline Execution

Execute the modular pipeline sequentially from terminal:

```bash
# Step 1: Run Exploratory Data Analysis (generates figures in reports/figures/)
python src/eda.py

# Step 2: Run Data Preprocessing (generates data/processed/model_ready.csv & models/scaler.pkl)
python -m src.preprocessing

# Step 3: Run Feature Engineering (generates data/processed/model_ready_features.csv)
python -m src.features

# Step 4: Train Models & Run Cross-Validation Benchmark (saves models & metrics)
python -m src.models
```

---

## 🚀 Interactive Streamlit Dashboard

Launch the decision-support web application:

```bash
streamlit run dashboard/app.py
```

Access the dashboard at **`http://localhost:8501`**.

### Dashboard Modules:
- **🎛️ Scenario Simulation Control Sidebar:** Live sliders for seasonal rainfall ($200-3000\text{ mm}$), temperature ($20-40^\circ\text{C}$), real GDP ($1-120\text{ B}\$$), inflation ($0-100\%$), acreage, and seasonal identity.
- **🤖 Real-Time Forecast KPI Card:** Real-time yield predictions in total Metric Tons, dynamic season badges, and percentage comparison against the 75-year historical seasonal baseline.
- **🔍 Tab 1 — Correlation Heatmap:** Interactive multi-variate Pearson correlation matrix.
- **📊 Tab 2 — Seasonal Distribution Analysis:** Overlay histograms and land extent scatter plots.
- **🌦️ Tab 3 — Decadal Climate Outliers:** Box plots isolating extreme precipitation and temperature anomalies.
- **🏆 Tab 4 — Feature Importance Ranking:** Gini and split importance weight rankings.
- **📋 Model Benchmark Summary:** Comparative evaluation table across all 4 architectures.

---

## 👥 Academic Metadata & Group Contributions

- **Module:** IT41033 — Nature Inspired Algorithms (NIA) Mini-Project
- **Institution:** Faculty of Information Technology, Horizon Campus
- **Module Coordinator:** Mr. Sanka Wijewardene
- **Submission Date:** 19 July 2026

| Student Name | Student ID | Designated Project Role | Core Deliverables |
|---|---|---|---|
| **E. Dasun Manjitha** | ITBIN-2313-0062 | **Pipeline Architecture Lead** | Project scaffold, data loader module (`src/data_loader.py`), Streamlit dashboard architecture (`dashboard/app.py`). |
| **W.G.C.M. Nimsara** | ITBIN-2313-0072 | **Data Quality & Preprocessing Specialist** | Cleaning engine, distribution-aware imputation, IQR winsorization, train-isolated `StandardScaler` (`src/preprocessing.py`, `notebooks/02_`). |
| **R.T. Dinith Sasanga** | ITBIN-2313-0101 | **Categorical Engineering Specialist** | Seasonal binarization ($S_{binary}$), autoregressive lags ($t-1, t-2$), 3-season rolling climate means, interaction features (`src/features.py`, `notebooks/03_`). |
| **W.A.S.I. Wijesinghe** | ITBIN-2313-0129 | **Machine Learning Modeling Engineer** | `TimeSeriesSplit` cross-validation, tuning Random Forest, LightGBM, Linear Regression, and MLP Neural Network, evaluation metrics (`src/models.py`, `notebooks/04_`, `tests/`). |
| **R.G.D.N. Wijesuriya** | ITBIN-2313-0130 | **Descriptive Statistics & Visualization Lead** | Exploratory Data Analysis, generating 8 diagnostic figures in `reports/figures/`, drafting reporting graphics & final report (`src/eda.py`, `notebooks/01_`). |

---

## 📜 References (IEEE Format)

```
[1] J. W. Jones et al., "The DSSAT cropping system model," European Journal of Agronomy, vol. 18, no. 3-4, pp. 235-265, 2003.
[2] D. B. Lobell and M. B. Burke, "Why are agricultural impacts of climate change so uncertain? The importance of temperature relative to precipitation," Environmental Research Letters, vol. 3, no. 3, p. 034007, 2008.
[3] T. Van Klompenburg, A. Kassahun, and C. Catal, "Crop yield prediction using machine learning: A systematic literature review," Computers and Electronics in Agriculture, vol. 177, p. 105709, 2020.
[4] Department of Census and Statistics Sri Lanka, "Paddy Statistics and Agricultural Extent Historical Reports (1952-2024)," Ministry of Finance, Economic Stabilization and National Policies, Colombo, Sri Lanka, 2024.
[5] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?," Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 507-520, 2022.
[6] Central Bank of Sri Lanka, "Annual Economic Review and Historical Time Series Tables (1950-2024)," CBSL Communications Department, Colombo, Sri Lanka, 2024.
[7] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 3146-3154, 2017.
[8] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.
```
