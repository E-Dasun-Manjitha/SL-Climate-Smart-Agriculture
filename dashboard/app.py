"""
Interactive Dashboard for Bi-Seasonal Paddy Yield Forecasting System
Sri Lanka Climate-Smart Agriculture (1950 - 2024)

Layout Architecture:
  1. Header & System Overview
  2. Sidebar Control Panel (Scenario Inputs & Sliders)
  3. Main Display: Real-Time Prediction KPI Card & Delta Baseline
  4. 4 Analytical Diagnostic Tabs:
     - Tab 1: Feature Correlation Heatmap
     - Tab 2: Seasonal Distribution Analysis (Histograms & Comparisons)
     - Tab 3: Climate Outliers & Extreme Anomalies (Decadal Box-Plots)
     - Tab 4: Feature Importance Ranking (Tree Ensembles)
  5. Model Performance & Evaluation Summary Footer
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration & Custom CSS Theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paddy Yield Forecasting System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1b4d3e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-caption {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .kpi-container {
        background: linear-gradient(135deg, #1b4d3e 0%, #2e7d32 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        margin-bottom: 1.8rem;
    }
    .kpi-title {
        font-size: 1.0rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .season-badge-maha {
        background-color: #1565c0;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .season-badge-yala {
        background-color: #e65100;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .footer-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_features.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ALL_MODELS_PATH = PROJECT_ROOT / "models" / "all_models.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['decade'] = (df['year'] // 10) * 10
    df['decade_label'] = df['decade'].astype(str) + "s"
    return df

@st.cache_resource
def load_models():
    best_pkg = joblib.load(BEST_MODEL_PATH)
    all_pkg = joblib.load(ALL_MODELS_PATH)
    scaler_pkg = joblib.load(SCALER_PATH)
    return best_pkg, all_pkg, scaler_pkg

df = load_data()
best_pkg, all_pkg, scaler_pkg = load_models()
best_model = best_pkg['model']
best_model_name = best_pkg['model_name']
metrics_df = all_pkg['metrics_df']

# ──────────────────────────────────────────────────────────────────────────────
# 1. Header & System Overview
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌾 Climate-Smart Agriculture: Bi-Seasonal Paddy Yield Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Bi-Seasonal Crop Forecasting using Hybrid Machine Learning (Random Forest & LightGBM) and Macroeconomic Indicators (1950 – 2024)</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Interactive Input Sidebar (Control Panel)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Scenario Simulation Panel")
st.sidebar.caption("Adjust environmental and economic factors to simulate harvest output:")

# Active Model Selector
selected_model_name = st.sidebar.selectbox(
    "Active Predictive Model",
    options=list(all_pkg['models'].keys()),
    index=0
)
active_model = all_pkg['models'][selected_model_name]

# Season Selector
season_choice = st.sidebar.radio(
    "Cultivation Season",
    options=["Maha (Major Monsoon)", "Yala (Inter-Monsoon)"],
    index=0
)
is_maha = "Maha" in season_choice
season_binary = 1 if is_maha else 0

st.sidebar.markdown("### 🌦️ Agrometeorological Drivers")
rainfall_input = st.sidebar.slider(
    "Seasonal Rainfall (mm)",
    min_value=200.0,
    max_value=3000.0,
    value=1450.0 if is_maha else 750.0,
    step=10.0,
    help="Cumulative precipitation across the cultivation season."
)

temp_input = st.sidebar.slider(
    "Mean Seasonal Temperature (°C)",
    min_value=20.0,
    max_value=40.0,
    value=26.5 if is_maha else 28.8,
    step=0.1,
    help="Mean surface temperature during crop vegetative and ripening phases."
)

st.sidebar.markdown("### 📈 Macroeconomic & Input Parameters")
gdp_input = st.sidebar.slider(
    "Real GDP Level (Billion USD)",
    min_value=1.0,
    max_value=120.0,
    value=85.0,
    step=1.0,
    help="Macroeconomic capacity indicator."
)

inflation_input = st.sidebar.slider(
    "Seasonal Inflation Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=5.5,
    step=0.5,
    help="Consumer Price Inflation index reflecting input cost dynamics."
)

st.sidebar.markdown("### 🌾 Acreage Estimates")
sown_input = st.sidebar.slider(
    "Sown Extent (000 Acres)",
    min_value=200.0,
    max_value=2000.0,
    value=1350.0 if is_maha else 850.0,
    step=10.0
)

harvest_ratio = st.sidebar.slider(
    "Harvested-to-Sown Area Ratio",
    min_value=0.60,
    max_value=1.00,
    value=0.95,
    step=0.01,
    help="Ratio of planted acreage successfully harvested."
)
harvested_input = sown_input * harvest_ratio

# Previous Season Lag (Autoregression)
historical_season_mean = df[df['season'] == ('Maha' if is_maha else 'Yala')]['target_yield_production'].mean()
yield_lag1_input = st.sidebar.number_input(
    "Previous Season Production (000 Mt)",
    min_value=100.0,
    max_value=3500.0,
    value=float(df['target_yield_production'].iloc[-1]),
    step=25.0
)

# ──────────────────────────────────────────────────────────────────────────────
# Prediction Computation
# ──────────────────────────────────────────────────────────────────────────────
scaler = scaler_pkg['scaler']
raw_continuous = np.array([[rainfall_input, temp_input, gdp_input, inflation_input, sown_input, harvested_input]])
scaled_vals = scaler.transform(raw_continuous)[0]

rain_s, temp_s, gdp_s, inf_s, sown_s, harv_s = scaled_vals
rain_x_temp = rain_s * temp_s
inf_x_gdp = inf_s * gdp_s
drought_flag = 1 if (is_maha and rainfall_input < 900) or (not is_maha and rainfall_input < 500) else 0

feat_vector = {
    'season_binary': season_binary,
    'rainfall_mm_scaled': rain_s,
    'temperature_c_scaled': temp_s,
    'gdp_billion_usd_scaled': gdp_s,
    'inflation_pct_scaled': inf_s,
    'sown_000_acres_scaled': sown_s,
    'harvested_000_acres_scaled': harv_s,
    'yield_lag1': yield_lag1_input,
    'yield_lag2': yield_lag1_input,
    'rainfall_roll3': rainfall_input,
    'temp_roll3': temp_input,
    'rain_x_temp': rain_x_temp,
    'inf_x_gdp': inf_x_gdp,
    'drought_flag': drought_flag,
    'harvest_efficiency': harvest_ratio
}

X_input = pd.DataFrame([feat_vector])[best_pkg['feature_cols']]
pred_output_000_mt = float(active_model.predict(X_input)[0])
pred_output_mt = pred_output_000_mt * 1000.0

delta_pct = ((pred_output_000_mt - historical_season_mean) / historical_season_mean) * 100.0
delta_symbol = "+" if delta_pct >= 0 else ""

# ──────────────────────────────────────────────────────────────────────────────
# 3. Main Display Area: Real-Time Prediction KPI Card
# ──────────────────────────────────────────────────────────────────────────────
badge_class = "season-badge-maha" if is_maha else "season-badge-yala"
season_name = "MAHA SEASON (North-East Monsoon)" if is_maha else "YALA SEASON (South-West Monsoon)"

st.markdown(f"""
<div class="kpi-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="kpi-title">🤖 Real-Time Machine Learning Yield Forecast</span>
        <span class="{badge_class}">{season_name}</span>
    </div>
    <div class="kpi-value">{pred_output_mt:,.2f} Metric Tons</div>
    <div style="font-size: 1.15rem; font-weight: 500;">
        ≈ <b>{pred_output_000_mt:,.1f}</b> Thousand Metric Tons &nbsp;|&nbsp; 
        <b>{delta_symbol}{delta_pct:.1f}%</b> relative to 75-year historical {('Maha' if is_maha else 'Yala')} average ({historical_season_mean:,.0f}k Mt)
    </div>
</div>
""", unsafe_allow_html=True)

# 3 KPI Sub-metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Estimated Yield Density", f"{pred_output_mt / (harvested_input * 1000) * 1000:,.1f} kg/Acre")
with c2:
    st.metric("Harvesting Land Efficiency", f"{harvest_ratio*100:.1f}%", f"{harvested_input:,.0f}k Harvested Acres")
with c3:
    st.metric("Active Model Engine", selected_model_name)
with c4:
    if drought_flag:
        st.metric("Climate Risk Status", "⚠️ Severe Drought Anomaly", "-1.0 Std Deviation", delta_color="inverse")
    else:
        st.metric("Climate Risk Status", "✅ Favorable Precipitation", "Optimal Range")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Analytical Diagnostic Tabs (Visual Analytics)
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Feature Correlation Heatmap",
    "📊 Seasonal Distribution Analysis",
    "🌦️ Climate Outliers & Decadal Anomalies",
    "🏆 Feature Importance Ranking"
])

# ── Tab 1: Feature Correlation Heatmap ──
with tab1:
    st.subheader("🔍 Multi-Variate Pearson Correlation Matrix")
    st.write("Visualizing correlations between climatic drivers, macroeconomic factors, cultivation acreage, and seasonal production yields.")
    
    corr_cols = [
        'target_yield_production', 'rainfall_mm', 'temperature_c',
        'gdp_billion_usd', 'inflation_pct', 'sown_000_acres', 'harvested_000_acres',
        'yield_lag1', 'rainfall_roll3', 'rain_x_temp'
    ]
    labels_clean = [
        'Production (Mt)', 'Rainfall (mm)', 'Temperature (°C)',
        'GDP ($B)', 'Inflation (%)', 'Sown (Acres)', 'Harvested (Acres)',
        'Yield Lag (t-1)', 'Rainfall Roll(3)', 'Rain × Temp'
    ]
    
    corr_matrix = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        x=labels_clean,
        y=labels_clean,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Multi-Decadal Feature Correlation Matrix (1950 - 2024)"
    )
    fig_corr.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 2: Seasonal Distribution Analysis ──
with tab2:
    st.subheader("📊 Seasonal Production Volume Distributions & Extent Dynamics")
    col_t2_1, col_t2_2 = st.columns(2)
    
    with col_t2_1:
        fig_hist = px.histogram(
            df, x='target_yield_production', color='season', barmode='overlay',
            color_discrete_map={'Maha': '#1565c0', 'Yala': '#e65100'},
            title="Production Volume Distribution by Season (Maha vs Yala)",
            labels={'target_yield_production': 'Production (000 Mt)', 'season': 'Season'},
            marginal='box', opacity=0.75
        )
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_t2_2:
        fig_scatter = px.scatter(
            df, x='harvested_000_acres', y='target_yield_production', color='season',
            size='rainfall_mm', hover_data=['year'],
            color_discrete_map={'Maha': '#1565c0', 'Yala': '#e65100'},
            title="Harvested Land Extent vs Output Volume (Circle size = Rainfall)",
            labels={'harvested_000_acres': 'Harvested Extent (000 Acres)', 'target_yield_production': 'Paddy Production (000 Mt)'}
        )
        fig_scatter.update_layout(template='plotly_white')
        st.plotly_chart(fig_scatter, use_container_width=True)

# ── Tab 3: Climate Outliers & Extreme Anomalies ──
with tab3:
    st.subheader("🌦️ Historical Climate Variance & Decadal Outliers")
    st.write("Box plots isolating extreme climatic outliers (droughts and temperature anomalies) across 7 decades.")
    
    metric_box = st.selectbox(
        "Select Indicator for Decadal Outlier Analysis:",
        ['rainfall_mm', 'temperature_c', 'inflation_pct', 'gdp_billion_usd'],
        format_func=lambda x: {
            'rainfall_mm': 'Seasonal Precipitation (mm)',
            'temperature_c': 'Surface Temperature (°C)',
            'inflation_pct': 'Consumer Price Inflation (%)',
            'gdp_billion_usd': 'Gross Domestic Product (Billion USD)'
        }[x]
    )
    
    fig_box = px.box(
        df, x='decade_label', y=metric_box, color='season',
        color_discrete_map={'Maha': '#1565c0', 'Yala': '#e65100'},
        title=f"Decadal Box-Plot Distribution & Outlier Identification — {metric_box.replace('_', ' ').title()}",
        labels={'decade_label': 'Decade', metric_box: metric_box.replace('_', ' ').title()},
        points='outliers'
    )
    fig_box.update_layout(template='plotly_white', boxmode='group', height=550)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 4: Feature Importance Ranking ──
with tab4:
    st.subheader("🏆 Machine Learning Feature Importance & Attribution Ranking")
    st.write("Relative feature importance weights extracted from the trained ensemble models:")
    
    col_f1, col_f2 = st.columns(2)
    rf_model = all_pkg['models']['Random Forest']
    lgb_model = all_pkg['models']['LightGBM']
    
    feat_names_readable = [
        'Season (Maha/Yala)', 'Rainfall (scaled)', 'Temperature (scaled)',
        'GDP Level (scaled)', 'Inflation (scaled)', 'Sown Area (scaled)',
        'Harvested Area (scaled)', 'Yield Lag 1 (t-1)', 'Yield Lag 2 (t-2)',
        'Rainfall 3-Season Roll', 'Temp 3-Season Roll', 'Rain × Temp',
        'Inflation × GDP', 'Drought Anomaly Flag', 'Harvesting Efficiency'
    ]
    
    with col_f1:
        rf_df = pd.DataFrame({
            'Feature': feat_names_readable,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_rf = px.bar(
            rf_df, x='Importance', y='Feature', orientation='h',
            title="Random Forest Gini Feature Importance",
            color_discrete_sequence=['#2e7d32']
        )
        fig_rf.update_layout(template='plotly_white')
        st.plotly_chart(fig_rf, use_container_width=True)
        
    with col_f2:
        lgb_df = pd.DataFrame({
            'Feature': feat_names_readable,
            'Importance': lgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_lgb = px.bar(
            lgb_df, x='Importance', y='Feature', orientation='h',
            title="LightGBM Split Weight Feature Importance",
            color_discrete_sequence=['#1565c0']
        )
        fig_lgb.update_layout(template='plotly_white')
        st.plotly_chart(fig_lgb, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Model Performance & Evaluation Summary Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Predictive Model Validation & Empirical Benchmark Summary")

col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    model_row = metrics_df[metrics_df['Model'] == selected_model_name].iloc[0]
    st.markdown(f"""
    <div class="footer-box">
        <h4 style="margin-top:0; color:#1b4d3e;">Active Model Performance: {selected_model_name}</h4>
        <p><b>Holdout Partition:</b> 2013 – 2024 (15% Chronological)</p>
        <hr style="margin: 8px 0;">
        <p><b>Test R² Score:</b> <span style="color:#2e7d32; font-weight:700;">{model_row['Test R²']:.4f}</span></p>
        <p><b>Test RMSE:</b> ±{model_row['Test RMSE (000 Mt)']*1000:,.0f} Metric Tons ({model_row['Test RMSE (000 Mt)']:.2f}k Mt)</p>
        <p><b>Test MAE:</b> ±{model_row['Test MAE (000 Mt)']*1000:,.0f} Metric Tons ({model_row['Test MAE (000 Mt)']:.2f}k Mt)</p>
        <p><b>Train R²:</b> {model_row['Train R²']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.dataframe(
        metrics_df.style.format({
            'Test RMSE (000 Mt)': '{:.2f}',
            'Test MAE (000 Mt)': '{:.2f}',
            'Test R²': '{:.3f}',
            'Train R²': '{:.3f}',
            'Generalization Gap (RMSE)': '{:.2f}'
        }).highlight_max(subset=['Test R²'], color='#c8e6c9')
          .highlight_min(subset=['Test RMSE (000 Mt)', 'Test MAE (000 Mt)'], color='#c8e6c9'),
        use_container_width=True
    )

st.caption("IT41033 Nature Inspired Algorithms (NIA) | Faculty of IT, Horizon Campus | Group Mini-Project (2026)")
