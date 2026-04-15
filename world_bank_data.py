"""
World Bank data fetcher for Pakistan Energy Demand Forecasting project.
Fetches GDP and population data for Pakistan from World Bank API.
"""

import pandas as pd
import requests
import time
from pathlib import Path


PAKISTAN_WB_CODE = "PAK"

GDP_INDICATOR = "NY.GDP.MKTP.CD"
POPULATION_INDICATOR = "SP.POP.TOTL"


def fetch_world_bank_data(
    indicator: str,
    country: str = PAKISTAN_WB_CODE,
    start_year: int = 1995,
    end_year: int = 2030
) -> pd.DataFrame:
    """
    Fetch World Bank indicator data for Pakistan.
    
    Args:
        indicator: World Bank indicator code (e.g., 'NY.GDP.MKTP.CD' for GDP)
        country: Country ISO code (default: 'PAK' for Pakistan)
        start_year: Start year for data
        end_year: End year for data
    
    Returns:
        DataFrame with year and value columns
    """
    base_url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 100
    }
    
    all_data = []
    page = 1
    total_pages = 1
    
    while page <= total_pages:
        params["page"] = page
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            
            if len(json_data) < 2 or json_data[1] is None:
                break
            
            total_pages = json_data[0].get('pages', 1)
            page_data = json_data[1]
            
            for entry in page_data:
                if entry['value'] is not None:
                    all_data.append({
                        'year': int(entry['date']),
                        'value': float(entry['value'])
                    })
            
            page += 1
            time.sleep(0.25)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {indicator}: {e}")
            break
    
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df = df.sort_values('year').reset_index(drop=True)
    
    return df


def fetch_gdp_data(start_year: int = 1995, end_year: int = 2030) -> pd.DataFrame:
    """Fetch GDP data for Pakistan."""
    return fetch_world_bank_data(
        indicator=GDP_INDICATOR,
        start_year=start_year,
        end_year=end_year
    )


def fetch_population_data(start_year: int = 1995, end_year: int = 2030) -> pd.DataFrame:
    """Fetch population data for Pakistan."""
    return fetch_world_bank_data(
        indicator=POPULATION_INDICATOR,
        start_year=start_year,
        end_year=end_year
    )


def fetch_all_world_bank_data(
    start_year: int = 1995,
    end_year: int = 2030,
    save: bool = True
) -> pd.DataFrame:
    """
    Fetch all World Bank data for Pakistan.
    
    Args:
        start_year: Start year for data
        end_year: End year for data  
        save: Whether to save to CSV
    
    Returns:
        DataFrame with GDP and population data
    """
    print("Fetching GDP data from World Bank...")
    gdp_df = fetch_gdp_data(start_year, end_year)
    gdp_df = gdp_df.rename(columns={'value': 'gdp_usd'})
    gdp_df['gdp_usd'] = gdp_df['gdp_usd'] / 1e9
    gdp_df = gdp_df.rename(columns={'gdp_usd': 'gdp_billion_usd'})
    
    print("Fetching population data from World Bank...")
    pop_df = fetch_population_data(start_year, end_year)
    pop_df = pop_df.rename(columns={'value': 'population'})
    pop_df['population'] = pop_df['population'] / 1e6
    pop_df = pop_df.rename(columns={'population': 'population_millions'})
    
    df = pd.merge(gdp_df, pop_df, on='year', how='outer')
    df = df.sort_values('year').reset_index(drop=True)
    
    if save:
        output_path = Path("data/raw/wbg_data.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"World Bank data saved to: {output_path}")
    
    return df


def load_world_bank_data(filepath: str = "data/raw/wbg_data.csv") -> pd.DataFrame:
    """Load World Bank data from CSV, or fetch if not available."""
    path = Path(filepath)
    
    if path.exists():
        df = pd.read_csv(path)
        return df
    
    print(f"World Bank data not found at {filepath}. Fetching...")
    return fetch_all_world_bank_data()


if __name__ == "__main__":
    df = fetch_all_world_bank_data(2000, 2024)
    print(f"\nLoaded {len(df)} rows of World Bank data")
    print(df.tail(10))