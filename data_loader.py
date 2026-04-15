"""
Data loader for Pakistan Energy Demand Forecasting project.
Loads demand data from the original Pakistan Energy Pipeline database.
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path


def get_db_path():
    """Get path to the original pipeline database."""
    current_dir = Path(__file__).parent
    
    db_paths = [
        current_dir / "db" / "pakistan_energy.db",
        current_dir.parent / "db" / "pakistan_energy.db",
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            return str(db_path)
    
    raise FileNotFoundError(
        f"Database not found. Expected at: {db_paths}. "
        "Please ensure the original pipeline database exists."
    )


def load_demand_data(period_start: int = 2000, period_end: int = 2024) -> pd.DataFrame:
    """
    Load electricity demand data from the original database.
    
    Args:
        period_start: Start year for data loading
        period_end: End year for data loading
    
    Returns:
        DataFrame with demand_twh and related features
    """
    db_path = get_db_path()
    
    query = """
    SELECT 
        year,
        demand_twh,
        demand_mwh_per_capita,
        capacity_mw,
        primary_energy_twh,
        gen_solar_twh,
        gen_wind_twh,
        gen_hydro_twh,
        gen_gas_twh,
        gen_coal_twh,
        gen_nuclear_twh,
        share_solar_pct,
        share_wind_pct,
        share_hydro_pct
    FROM energy_data
    WHERE year BETWEEN ? AND ?
    ORDER BY year
    """
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=(period_start, period_end))
    conn.close()
    
    df = df.copy()
    df['year'] = df['year'].astype(int)
    
    return df


def load_all_energy_data(period_start: int = 2000, period_end: int = 2024) -> pd.DataFrame:
    """
    Load all available energy data from the original database.
    
    Args:
        period_start: Start year for data loading
        period_end: End year for data loading
    
    Returns:
        DataFrame with all energy columns
    """
    db_path = get_db_path()
    
    query = """
    SELECT *
    FROM energy_data
    WHERE year BETWEEN ? AND ?
    ORDER BY year
    """
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn, params=(period_start, period_end))
    conn.close()
    
    df = df.copy()
    df['year'] = df['year'].astype(int)
    
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")


if __name__ == "__main__":
    df = load_demand_data(2000, 2024)
    print(f"Loaded {len(df)} rows of demand data")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nDemand range: {df['demand_twh'].min():.2f} - {df['demand_twh'].max():.2f} TWh")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")

    save_data(df, "data/processed/demand_data.csv")