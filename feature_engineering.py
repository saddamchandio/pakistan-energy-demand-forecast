"""
Feature engineering module for Pakistan Energy Demand Forecasting project.
Merges demand data with World Bank features and creates engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_demand_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load demand data from database or CSV."""
    if filepath and Path(filepath).exists():
        return pd.read_csv(filepath)
    
    from data_loader import load_demand_data
    return load_demand_data(2000, 2024)


def load_world_bank_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load World Bank data from CSV."""
    if filepath is None:
        filepath = "data/raw/wbg_data.csv"
    
    path = Path(filepath)
    
    if not path.exists():
        from world_bank_data import fetch_all_world_bank_data
        return fetch_all_world_bank_data(2000, 2024, save=True)
    
    return pd.read_csv(path)


def merge_datasets(
    demand_df: pd.DataFrame,
    wbg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge demand data with World Bank data.
    
    Args:
        demand_df: DataFrame with demand data
        wbg_df: DataFrame with World Bank data (GDP, population)
    
    Returns:
        Merged DataFrame
    """
    demand_df = demand_df.copy()
    wbg_df = wbg_df.copy()
    
    demand_df = demand_df.drop(columns=[
        'demand_mwh_per_capita'
    ], errors='ignore')
    
    wbg_cols = {'year': 'year'}
    if 'gdp_billion_usd' in wbg_df.columns:
        wbg_cols['gdp_billion_usd'] = 'gdp_billion_usd'
    if 'population_millions' in wbg_df.columns:
        wbg_cols['population_millions'] = 'population_millions'
    
    wbg_df = wbg_df[list(wbg_cols.keys())]
    wbg_df = wbg_df.rename(columns=wbg_cols)
    
    df = pd.merge(demand_df, wbg_df, on='year', how='left')
    
    df = df.sort_values('year').reset_index(drop=True)
    
    return df


def create_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Create lag features for demand.
    
    Args:
        df: DataFrame with demand_twh column
        lags: List of lag periods
    
    Returns:
        DataFrame with additional lag columns
    """
    df = df.copy()
    
    for lag in lags:
        df[f'demand_twh_lag_{lag}'] = df['demand_twh'].shift(lag)
    
    return df


def create_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create growth rate features.
    
    Args:
        df: DataFrame with demand_twh column
    
    Returns:
        DataFrame with growth features
    """
    df = df.copy()
    
    df['demand_growth_yoy'] = df['demand_twh'].pct_change() * 100
    
    df['demand_growth_3yr'] = df['demand_twh'].pct_change(periods=3) * 100
    
    df['demand_ma_3yr'] = df['demand_twh'].rolling(window=3, min_periods=1).mean()
    
    df['demand_ma_5yr'] = df['demand_twh'].rolling(window=5, min_periods=1).mean()
    
    if 'gdp_billion_usd' in df.columns:
        df['gdp_growth_yoy'] = df['gdp_billion_usd'].pct_change() * 100
    
    if 'population_millions' in df.columns:
        df['pop_growth_yoy'] = df['population_millions'].pct_change() * 100
    
    return df


def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """Calculate Compound Annual Growth Rate."""
    if start_value <= 0 or end_value <= 0:
        return 0.0
    return ((end_value / start_value) ** (1 / periods) - 1) * 100


def add_cagr_column(df: pd.DataFrame, column: str, window: int = 5) -> pd.DataFrame:
    """
    Add rolling CAGR for a given column.
    
    Args:
        df: DataFrame
        column: Column name to calculate CAGR
        window: Number of years for CAGR calculation
    
    Returns:
        DataFrame with CAGR column
    """
    df = df.copy()
    
    cagr_values = []
    for i in range(len(df)):
        if i < window:
            cagr_values.append(None)
        else:
            start_val = df[column].iloc[i - window]
            end_val = df[column].iloc[i]
            cagr = calculate_cagr(start_val, end_val, window)
            cagr_values.append(cagr)
    
    df[f'{column}_cagr_{window}yr'] = cagr_values
    
    return df


def engineer_features(
    demand_df: pd.DataFrame,
    wbg_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Args:
        demand_df: DataFrame with demand data
        wbg_df: DataFrame with World Bank data (optional)
    
    Returns:
        Feature-engineered DataFrame
    """
    if wbg_df is not None:
        df = merge_datasets(demand_df, wbg_df)
    else:
        df = demand_df.copy()
    
    df = create_lag_features(df, lags=[1, 2, 3])
    
    df = create_growth_features(df)
    
    df = add_cagr_column(df, 'demand_twh', window=5)
    
    df = df.dropna()
    
    return df


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Prophet model.
    
    Args:
        df: Feature-engineered DataFrame
    
    Returns:
        DataFrame with 'ds' and 'y' columns, plus regressors
    """
    prophet_df = pd.DataFrame()
    
    prophet_df['ds'] = pd.to_datetime(df['year'], format='%Y')
    prophet_df['y'] = df['demand_twh']
    
    if 'gdp_billion_usd' in df.columns:
        prophet_df['gdp_billion_usd'] = df['gdp_billion_usd'].fillna(method='ffill')
    
    if 'capacity_mw' in df.columns:
        prophet_df['capacity_mw'] = df['capacity_mw']
    
    return prophet_df


def prepare_arima_data(df: pd.DataFrame) -> pd.Series:
    """
    Prepare data for ARIMA model.
    
    Args:
        df: Feature-engineered DataFrame
    
    Returns:
        Series with demand_twh indexed by year
    """
    arima_series = df.set_index('year')['demand_twh']
    
    return arima_series


def get_feature_importance_summary(df: pd.DataFrame) -> dict:
    """Get summary of features available for modeling."""
    exclude_cols = ['year', 'demand_twh']
    
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols
    ]
    
    summary = {
        'total_features': len(feature_cols),
        'features': feature_cols,
        'data_range': f"{df['year'].min()}-{df['year'].max()}",
        'observations': len(df)
    }
    
    return summary


if __name__ == "__main__":
    demand_df = load_demand_data()
    print(f"Loaded demand data: {len(demand_df)} rows")
    
    wbg_df = load_world_bank_data()
    print(f"Loaded World Bank data: {len(wbg_df)} rows")
    
    df = engineer_features(demand_df, wbg_df)
    print(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
    print(f"\nFeatures: {df.columns.tolist()}")
    
    output_path = Path("data/processed/feature_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")