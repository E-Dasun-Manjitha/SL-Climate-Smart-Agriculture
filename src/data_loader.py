"""
Data Loader Module for Bi-Seasonal Paddy Yield Forecasting Pipeline.

Loads and consolidates raw datasets from data/raw/ into unified DataFrames.
Handles header variations, multi-file concatenation, and schema normalization
for agriculture, climate, and macroeconomic data sources.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ──────────────────────────────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ══════════════════════════════════════════════════════════════════════════════
#  1. PRIMARY DATASET: Kaggle Sri Lanka Rice Production & Climate
# ══════════════════════════════════════════════════════════════════════════════

def load_kaggle_rice_climate() -> pd.DataFrame:
    """
    Load the Kaggle Sri Lanka Rice Production & Climate dataset.
    
    This is the primary dataset containing national-level seasonal data 
    from 1950–2024 with columns: Year, Season, Sown, Harvested, GDP, 
    Inflation, Rainfall, Temperature, Production.
    
    Returns:
        pd.DataFrame with standardized column names.
    """
    filepath = RAW_DATA_DIR / "kaggle_rice_climate" / "rice new one.xlsx"
    df = pd.read_excel(filepath)
    
    # Standardize column names
    col_map = {
        'Year_New': 'year',
        'Season': 'season',
        'Sown (*000  Acres)': 'sown_000_acres',
        'Harvested (*000  Acres)': 'harvested_000_acres',
        'GDP B$': 'gdp_billion_usd',
        'Inflation(%)': 'inflation_pct',
        'Rainfall(mm)': 'rainfall_mm',
        'Temperature(°C)': 'temperature_c',
        'Production (*000  Mt.)': 'production_000_mt',
    }
    df = df.rename(columns=col_map)
    
    # Ensure correct dtypes
    df['year'] = df['year'].astype(int)
    df['season'] = df['season'].str.strip().str.title()
    
    for col in ['sown_000_acres', 'harvested_000_acres', 'gdp_billion_usd',
                'inflation_pct', 'rainfall_mm', 'temperature_c', 'production_000_mt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. DCS PADDY YIELD & PRODUCTION BY DISTRICT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dcs_district_wide_excel(filepath: Path, value_label: str) -> pd.DataFrame:
    """
    Parse DCS district-level Excel files with wide (years-as-columns) layout.
    
    These files have a complex header structure:
      Row 0: empty
      Row 1: title + value_label in last columns  
      Row 2: 'District' | year1 | ... | yearN  (years across columns, 4 cols each)
      Row 3: sub-headers: Major | Minor | Rainfed | Total (repeating per year)
      Row 4+: district data rows
    
    We extract only the 'Total' sub-column for each year and unpivot.
    
    Args:
        filepath: Path to the Excel file.
        value_label: Name for the value column (e.g., 'avg_yield_kg_per_ha').
    
    Returns:
        pd.DataFrame with columns: [district, year, season, <value_label>]
    """
    df_raw = pd.read_excel(filepath, sheet_name=0, header=None)
    
    # Determine season from filename
    fname = filepath.stem.lower()
    if 'maha' in fname:
        season = 'Maha'
    elif 'yala' in fname:
        season = 'Yala'
    else:
        season = 'Unknown'
    
    # Find the header rows: Row with 'District' and Row with 'Major'/'Minor'/'Total'
    district_row_idx = None
    for i in range(min(10, len(df_raw))):
        row_vals = [str(v).strip().lower() for v in df_raw.iloc[i].values if pd.notna(v)]
        if 'district' in row_vals:
            district_row_idx = i
            break
    
    if district_row_idx is None:
        raise ValueError(f"Cannot find 'District' header row in {filepath}")
    
    sub_header_row_idx = district_row_idx + 1
    
    # Extract year headers from the district row
    year_row = df_raw.iloc[district_row_idx]
    sub_row = df_raw.iloc[sub_header_row_idx]
    
    # Find columns where 'Total' appears in sub-header row
    total_col_indices = []
    year_labels = []
    
    for col_idx in range(len(sub_row)):
        val = str(sub_row.iloc[col_idx]).strip().lower()
        if val == 'total':
            total_col_indices.append(col_idx)
            # Find the corresponding year by scanning backwards in year_row
            year_val = None
            for scan_idx in range(col_idx, -1, -1):
                candidate = year_row.iloc[scan_idx]
                if pd.notna(candidate) and str(candidate).strip().lower() != 'nan':
                    candidate_str = str(candidate).strip()
                    if candidate_str.lower() != 'district':
                        year_val = candidate_str
                        break
            year_labels.append(year_val)
    
    # Data starts after sub-header row
    data_start = sub_header_row_idx + 1
    
    # Find district column (column with 'District' header, usually col 1)
    district_col_idx = None
    for col_idx in range(len(year_row)):
        if pd.notna(year_row.iloc[col_idx]) and str(year_row.iloc[col_idx]).strip().lower() == 'district':
            district_col_idx = col_idx
            break
    
    if district_col_idx is None:
        district_col_idx = 1  # Default fallback
    
    records = []
    for row_idx in range(data_start, len(df_raw)):
        district = df_raw.iloc[row_idx, district_col_idx]
        if pd.isna(district) or str(district).strip() == '':
            continue
        district = str(district).strip().upper()
        
        # Skip summary/total rows
        if district in ['SRI LANKA', 'TOTAL', 'ALL ISLAND', 'ISLAND', 'ISLAND TOTAL',
                        'ALL DISTRICTS', 'SOURCE', 'NOTE', 'NAN']:
            continue
        
        for ti, col_idx in enumerate(total_col_indices):
            year_str = year_labels[ti]
            if year_str is None:
                continue
            
            # Parse year: handle formats like '1978/1979', '2000/2021', '1979'
            try:
                if '/' in str(year_str):
                    # Take the first year for Maha (e.g., 1978/1979 -> 1978)
                    # or for consistency, take the first part
                    parts = str(year_str).split('/')
                    year = int(parts[0])
                else:
                    year = int(float(year_str))
            except (ValueError, TypeError):
                continue
            
            value = df_raw.iloc[row_idx, col_idx]
            value = pd.to_numeric(value, errors='coerce')
            
            records.append({
                'district': district,
                'year': year,
                'season': season,
                value_label: value
            })
    
    return pd.DataFrame(records)


def _parse_dcs_simple_wide_excel(filepath: Path, value_label: str) -> pd.DataFrame:
    """
    Parse DCS district-level Excel files with simple wide layout.
    
    These files (e.g., Production) have one column per year:
      Row 0-1: empty/title
      Row 2: 'DISTRICT' | year1 | year2 | ... | yearN
      Row 3+: district data rows
    
    Args:
        filepath: Path to the Excel file.
        value_label: Name for the value column (e.g., 'production_000_mt').
    
    Returns:
        pd.DataFrame with columns: [district, year, season, <value_label>]
    """
    df_raw = pd.read_excel(filepath, sheet_name=0, header=None)
    
    # Determine season from filename
    fname = filepath.stem.lower()
    if 'maha' in fname:
        season = 'Maha'
    elif 'yala' in fname:
        season = 'Yala'
    else:
        season = 'Unknown'
    
    # Find header row with 'DISTRICT'
    header_row_idx = None
    for i in range(min(10, len(df_raw))):
        row_vals = [str(v).strip().lower() for v in df_raw.iloc[i].values if pd.notna(v)]
        if 'district' in row_vals:
            header_row_idx = i
            break
    
    if header_row_idx is None:
        raise ValueError(f"Cannot find 'DISTRICT' header row in {filepath}")
    
    header_row = df_raw.iloc[header_row_idx]
    
    # Find district column and year columns
    district_col_idx = None
    year_cols = []  # (col_idx, year_int)
    
    for col_idx in range(len(header_row)):
        val = header_row.iloc[col_idx]
        if pd.isna(val):
            continue
        val_str = str(val).strip()
        if val_str.lower() == 'district':
            district_col_idx = col_idx
        else:
            # Try to parse as year
            try:
                if '/' in val_str:
                    year = int(val_str.split('/')[0])
                else:
                    year = int(float(val_str))
                if 1900 < year < 2100:
                    year_cols.append((col_idx, year))
            except (ValueError, TypeError):
                pass
    
    if district_col_idx is None:
        district_col_idx = 1
    
    data_start = header_row_idx + 1
    records = []
    
    for row_idx in range(data_start, len(df_raw)):
        district = df_raw.iloc[row_idx, district_col_idx]
        if pd.isna(district) or str(district).strip() == '':
            continue
        district = str(district).strip().upper()
        
        if district in ['SRI LANKA', 'TOTAL', 'ALL ISLAND', 'ISLAND', 'ISLAND TOTAL',
                        'ALL DISTRICTS', 'SOURCE', 'NOTE', 'NAN']:
            continue
        
        for col_idx, year in year_cols:
            value = df_raw.iloc[row_idx, col_idx]
            value = pd.to_numeric(value, errors='coerce')
            records.append({
                'district': district,
                'year': year,
                'season': season,
                value_label: value
            })
    
    return pd.DataFrame(records)


def load_dcs_paddy_yield_district() -> pd.DataFrame:
    """
    Load and consolidate DCS Paddy Average Yield and Production by District.
    
    Reads 4 files:
      - Average_Yeild-Maha_Season-1979-2025.xlsx  (wide, 4 sub-cols per year)
      - Average_Yeild-Yala_Season-1979-2025.xlsx   (wide, 4 sub-cols per year)
      - Production-Maha_Season-1979-2025.xlsx      (simple, 1 col per year)
      - Production-Yala_Season-1979-2025.xlsx       (simple, 1 col per year)
    
    Returns:
        pd.DataFrame with columns: [district, year, season, avg_yield_kg_per_ha, production_000_mt]
    """
    folder = RAW_DATA_DIR / "dcs_paddy_yield_district"
    
    yield_frames = []
    prod_frames = []
    
    for f in sorted(folder.glob("*.xlsx")):
        fname = f.stem.lower()
        if 'yeild' in fname or 'yield' in fname:
            # These have Major/Minor/Rainfed/Total sub-columns
            df = _parse_dcs_district_wide_excel(f, 'avg_yield_kg_per_ha')
            yield_frames.append(df)
        elif 'production' in fname:
            # These have simple one-column-per-year layout
            df = _parse_dcs_simple_wide_excel(f, 'production_000_mt')
            prod_frames.append(df)
    
    # Combine yield files (Yala + Maha)
    df_yield = pd.concat(yield_frames, ignore_index=True) if yield_frames else pd.DataFrame()
    
    # Combine production files (Yala + Maha)
    df_prod = pd.concat(prod_frames, ignore_index=True) if prod_frames else pd.DataFrame()
    
    # Merge yield and production on (district, year, season)
    if not df_yield.empty and not df_prod.empty:
        df = pd.merge(df_yield, df_prod, on=['district', 'year', 'season'], how='outer')
    elif not df_yield.empty:
        df = df_yield
    else:
        df = df_prod
    
    return df.sort_values(['year', 'season', 'district']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  3. DCS PADDY EXTENT SOWN & HARVESTED BY DISTRICT
# ══════════════════════════════════════════════════════════════════════════════

def load_dcs_paddy_extent_district() -> pd.DataFrame:
    """
    Load and consolidate DCS Paddy Extent Sown and Harvested by District.
    
    Reads 4 files (Sown × Yala/Maha, Harvested × Yala/Maha).
    
    Returns:
        pd.DataFrame with columns: [district, year, season, sown_ha, harvested_ha]
    """
    folder = RAW_DATA_DIR / "dcs_paddy_extent_district"
    
    sown_frames = []
    harvested_frames = []
    
    for f in sorted(folder.glob("*.xlsx")):
        fname = f.stem.lower()
        if 'sown' in fname:
            df = _parse_dcs_district_wide_excel(f, 'sown_ha')
            sown_frames.append(df)
        elif 'harvested' in fname:
            df = _parse_dcs_district_wide_excel(f, 'harvested_ha')
            harvested_frames.append(df)
    
    df_sown = pd.concat(sown_frames, ignore_index=True) if sown_frames else pd.DataFrame()
    df_harv = pd.concat(harvested_frames, ignore_index=True) if harvested_frames else pd.DataFrame()
    
    if not df_sown.empty and not df_harv.empty:
        df = pd.merge(df_sown, df_harv, on=['district', 'year', 'season'], how='outer')
    elif not df_sown.empty:
        df = df_sown
    else:
        df = df_harv
    
    return df.sort_values(['year', 'season', 'district']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  4. DCS PADDY EXTENT NATIONAL
# ══════════════════════════════════════════════════════════════════════════════

def load_dcs_paddy_extent_national() -> pd.DataFrame:
    """
    Load DCS National-level Paddy Extent, Yield, and Production.
    
    Reads 2 files (Maha + Yala season).
    Each file has columns: Year, Sown (000 Acres, 000 Ha), 
    Harvested (000 Acres, 000 Ha), Average Yield (Bushels/Acre, Kg/Ha),
    Production (000 Bushels, 000 Mt.).
    
    Returns:
        pd.DataFrame with columns: [year, season, sown_000_ha, harvested_000_ha,
                                     avg_yield_kg_per_ha, production_000_mt]
    """
    folder = RAW_DATA_DIR / "dcs_paddy_extent_national"
    frames = []
    
    for f in sorted(folder.glob("*.xlsx")):
        fname = f.stem.lower()
        season = 'Maha' if 'maha' in fname else 'Yala' if 'yala' in fname else 'Unknown'
        
        # Read raw, skip decorative header rows
        # Header structure: Row 3 has column group names, Row 4 has units
        # Data starts at Row 5
        df_raw = pd.read_excel(f, sheet_name=0, header=None)
        
        # Find data start: look for first row with a year value (e.g., 1951/52)
        data_start = None
        for i in range(len(df_raw)):
            val = df_raw.iloc[i, 1]  # Year is in column 1
            if pd.notna(val):
                val_str = str(val).strip()
                if '/' in val_str or (val_str.isdigit() and int(val_str) > 1900):
                    data_start = i
                    break
        
        if data_start is None:
            continue
        
        records = []
        for i in range(data_start, len(df_raw)):
            year_val = df_raw.iloc[i, 1]
            if pd.isna(year_val):
                continue
            
            year_str = str(year_val).strip()
            try:
                if '/' in year_str:
                    year = int(year_str.split('/')[0])
                else:
                    year = int(float(year_str))
            except (ValueError, TypeError):
                continue
            
            records.append({
                'year': year,
                'season': season,
                'sown_000_ha': pd.to_numeric(df_raw.iloc[i, 3], errors='coerce'),
                'harvested_000_ha': pd.to_numeric(df_raw.iloc[i, 5], errors='coerce'),
                'avg_yield_kg_per_ha': pd.to_numeric(df_raw.iloc[i, 7], errors='coerce'),
                'production_000_mt': pd.to_numeric(df_raw.iloc[i, 9], errors='coerce'),
            })
        
        frames.append(pd.DataFrame(records))
    
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df.sort_values(['year', 'season']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  5. MACROECONOMIC DATA (loaded separately at Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

def load_cbsl_ccpi() -> pd.DataFrame:
    """Load Colombo Consumer Price Index (CCPI) from CBSL data."""
    folder = RAW_DATA_DIR / "cbsl_ccpi"
    # File has empty name (.xlsx)
    files = list(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")
    
    filepath = files[0]
    df_raw = pd.read_excel(filepath, sheet_name=0, header=None, nrows=20)
    
    # Inspect and return raw for now — Phase 3 will parse properly
    return pd.read_excel(filepath, sheet_name=0)


def load_cbsl_ncpi() -> pd.DataFrame:
    """Load National Consumer Price Index (NCPI) from CBSL data."""
    filepath = RAW_DATA_DIR / "cbsl_ncpi" / "NCPI_and_NCPI_CORE_20260721_e.xlsx"
    return pd.read_excel(filepath, sheet_name=0)


def load_cbsl_gdp_expenditure() -> pd.DataFrame:
    """Load GDP (Expenditure Approach) from CBSL data."""
    filepath = RAW_DATA_DIR / "cbsl_gdp_expenditure" / "Aggregate_Demand_at_Constant_2015_Prices_20260630_e.xlsx"
    return pd.read_excel(filepath, sheet_name=0)


def load_cbsl_gdp_production() -> pd.DataFrame:
    """Load GDP (Production Approach) from CBSL data."""
    folder = RAW_DATA_DIR / "cbsl_gdp_production"
    files = list(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {folder}")
    return pd.read_excel(files[0], sheet_name=0)


# ══════════════════════════════════════════════════════════════════════════════
#  6. SUPPLEMENTAL CLIMATE DATA (optional)
# ══════════════════════════════════════════════════════════════════════════════

def load_srilanka_climate_supplemental() -> Optional[pd.DataFrame]:
    """
    Load the Sri Lanka Climate Data CSV (supplemental).
    Only used if EDA reveals gaps in the Kaggle dataset's climate columns.
    
    Contains: date, latitude, longitude, temperature_2m_max, temperature_2m_min,
              precipitation_sum
    """
    filepath = RAW_DATA_DIR / "srilanka_climate" / "Sri Lanka Climate Data" / "Sri_Lanka_Climate_Data.csv"
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_primary() -> Dict[str, pd.DataFrame]:
    """
    Load all four primary agriculture/climate datasets.
    
    Returns:
        dict with keys:
            'kaggle_rice_climate' -> National seasonal data (1950-2024)
            'dcs_paddy_yield_district' -> District-level yield & production (1979-2025)
            'dcs_paddy_extent_district' -> District-level sown & harvested extent (1979-2025)
            'dcs_paddy_extent_national' -> National-level extent, yield & production (1951-2024)
    """
    datasets = {}
    
    print("Loading primary datasets...")
    print("-" * 60)
    
    # 1. Kaggle Rice & Climate
    print("\n[1/4] Kaggle Sri Lanka Rice Production & Climate...")
    datasets['kaggle_rice_climate'] = load_kaggle_rice_climate()
    
    # 2. DCS Paddy Yield District
    print("[2/4] DCS Paddy Yield & Production by District...")
    datasets['dcs_paddy_yield_district'] = load_dcs_paddy_yield_district()
    
    # 3. DCS Paddy Extent District
    print("[3/4] DCS Paddy Extent Sown & Harvested by District...")
    datasets['dcs_paddy_extent_district'] = load_dcs_paddy_extent_district()
    
    # 4. DCS Paddy Extent National
    print("[4/4] DCS Paddy Extent National...")
    datasets['dcs_paddy_extent_national'] = load_dcs_paddy_extent_national()
    
    # Print summaries
    print("\n" + "=" * 60)
    print("DATASET SUMMARIES")
    print("=" * 60)
    
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")
        if 'season' in df.columns:
            print(f"  Season values: {df['season'].unique()}")
        if 'year' in df.columns:
            print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
        missing = df.isnull().sum()
        if missing.any():
            print(f"  Missing values:\n{missing[missing > 0].to_string()}")
        else:
            print(f"  Missing values: None")
        print(f"  First 3 rows:\n{df.head(3).to_string()}")
    
    return datasets


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    data = load_primary()
    print(f"\n\nLoaded {len(data)} datasets successfully.")
