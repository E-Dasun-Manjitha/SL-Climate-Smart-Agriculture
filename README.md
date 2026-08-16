# Climate-Smart Agriculture: Bi-Seasonal Paddy Yield Forecasting
### Using Hybrid Predictive Data Mining and Macroeconomic Regression

**Module Code:** IT41033 — Nature Inspired Algorithms (NIA)  
**Institution:** Faculty of Information Technology, Horizon Campus  
**Module Coordinator:** Mr. Sanka Wijewardene  
**Submission Date:** 19 July 2026  

---

## Project Overview
Rice cultivation is the economic backbone supporting food security and rural livelihoods across Sri Lanka. Agricultural yields fluctuate with monsoonal patterns (*Maha* and *Yala* seasons) and compounding macroeconomic shocks (inflation and GDP variations). 

This project implements an automated machine learning forecasting pipeline utilizing over 70 years (1950–2024) of multi-variate historical agrometeorological and economic records. It benchmarks advanced ensemble regression models (**Random Forest**, **LightGBM**) against baseline **Linear Regression** and **Deep Neural Networks (MLP)**, deployed with an interactive **Streamlit** dashboard.

---

## Project Structure
```
paddy-yield-forecast/
├── data/
│   ├── raw/                  # Raw untracked datasets (DCS, Kaggle, CBSL)
│   ├── interim/              # Partially cleaned datasets
│   └── processed/            # Final model-ready datasets (model_ready.csv, model_ready_features.csv)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & Consistency Validation
│   ├── 02_preprocessing.ipynb# Cleaning, IQR Winsorization, Binarization & Scaling
│   ├── 03_feature_engineering.ipynb # Lag generation, Rolling climate & Interactions
│   ├── 04_modeling.ipynb     # Model training with TimeSeriesSplit CV
│   └── 05_evaluation.ipynb   # Benchmarking, Error metrics & Feature attribution
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Ingestion for Excel & CSV data streams
│   ├── eda.py                # EDA generation script
│   ├── preprocessing.py      # End-to-end preprocessing pipeline
│   ├── features.py           # Domain feature engineering module
│   └── models.py             # Model training, cross-validation & evaluation
├── dashboard/
│   └── app.py                # Multi-page interactive Streamlit dashboard
├── reports/
│   ├── figures/              # 8 High-resolution diagnostic figures (PNG)
│   ├── final_report.md       # Comprehensive academic final report draft
│   └── individual_contribution_template.md # Group member contribution table
├── models/
│   ├── best_model.pkl        # Serialized best model (Random Forest)
│   ├── all_models.pkl        # Serialized benchmark suite
│   └── scaler.pkl            # Fitted StandardScaler
├── tests/
│   └── test_pipeline.py      # Automated pytest test suite
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd SL-Climate-Smart-Agriculture

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/
```

---

## Pipeline Execution Workflow

Run the modules in sequential order:

```bash
# Step 1: Run Data Loading & Exploratory Data Analysis (generates figures in reports/figures/)
python src/eda.py

# Step 2: Run Data Preprocessing (generates data/processed/model_ready.csv & models/scaler.pkl)
python -m src.preprocessing

# Step 3: Run Feature Engineering (generates data/processed/model_ready_features.csv)
python -m src.features

# Step 4: Run Model Training & Cross-Validation Benchmarking (saves models & metrics)
python -m src.models
```

---

## Interactive Analytical Dashboard

Launch the Streamlit web dashboard locally:

```bash
streamlit run dashboard/app.py
```

### Features included in the Dashboard:
- 📊 **Executive Overview**: High-level production metrics and 75-year national trajectory.
- 🌦️ **Climate & Decadal Outliers**: Box-plots analyzing extreme weather anomalies across decades.
- 🌾 **Bi-Seasonal Production Dynamics**: Maha vs Yala harvest distributions and acreage efficiency.
- 🔍 **Feature Correlation Heatmap**: Interactive multi-variate Pearson correlation matrix.
- 🤖 **ML Model Benchmarks**: Comparative evaluation ($R^2$, RMSE, MAE) of all 4 architectures.
- 🎯 **Yield Predictor Simulator**: Interactive "What-If" scenario forecaster using the best model.

---

## Benchmark Results (Holdout Test Set: 2013–2024)

| Model Architecture | Test RMSE (000 Mt) | Test MAE (000 Mt) | Test $R^2$ | Train $R^2$ |
|---|---|---|---|
| 🏆 **Random Forest Regressor** | **419.43** | 376.02 | **0.5767** | 0.9892 |
| **LightGBM Regressor** | 452.94 | 365.35 | **0.5064** | 0.9873 |
| **Baseline Linear Regression** | 496.27 | **330.76** | 0.4075 | 0.9793 |
| **Deep Neural Network (MLP)** | 584.15 | 505.49 | 0.1790 | 0.8901 |

---

## Team Members & Responsibilities
- **E. Dasun Manjitha** (ITBIN-2313-0062) — *Pipeline Architecture Lead*
- **W.G.C.M. Nimsara** (ITBIN-2313-0072) — *Data Quality & Preprocessing Specialist*
- **R.T. Dinith Sasanga** (ITBIN-2313-0101) — *Categorical Engineering Specialist*
- **W.A.S.I. Wijesinghe** (ITBIN-2313-0129) — *Machine Learning Modeling Engineer*
- **R.G.D.N. Wijesuriya** (ITBIN-2313-0130) — *Descriptive Statistics & Visualization Lead*
