"""
Prophet model implementation for Pakistan Energy Demand Forecasting.
Uses Meta's Prophet for time series forecasting with GDP regressor.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Prophet model.
    
    Args:
        df: DataFrame with 'year', 'demand_twh', and optionally 'gdp_billion_usd'
    
    Returns:
        DataFrame with 'ds' and 'y' columns
    """
    prophet_df = pd.DataFrame()
    
    prophet_df['ds'] = pd.to_datetime(df['year'].astype(str), format='%Y')
    prophet_df['y'] = df['demand_twh']
    
    return prophet_df


def train_prophet_model(
    df: pd.DataFrame,
    growth: str = 'linear',
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
    add_gdp_regressor: bool = True,
    forecasting_years: int = 6
) -> Tuple[Prophet, pd.DataFrame, dict]:
    """
    Train Prophet model on demand data.
    
    Args:
        df: DataFrame with 'year', 'demand_twh', and optionally 'gdp_billion_usd'
        growth: Growth type ('linear' or 'logistic')
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality  
        daily_seasonality: Include daily seasonality
        add_gdp_regressor: Add GDP as regressor
        forecasting_years: Number of years to forecast
    
    Returns:
        Tuple of (trained_model, forecast_df, metrics)
    """
    train_df = prepare_prophet_data(df)
    
    if add_gdp_regressor and 'gdp_billion_usd' in df.columns:
        train_df['gdp'] = df['gdp_billion_usd'].values
    
    model = Prophet(
        growth=growth,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        interval_width=0.95
    )
    
    if add_gdp_regressor and 'gdp_billion_usd' in df.columns:
        model.add_regressor('gdp')
    
    model.fit(train_df)
    
    future_years = forecasting_years + 5
    future = model.make_future_dataframe(periods=future_years, freq='YE')
    future['year'] = future['ds'].dt.year
    
    if add_gdp_regressor and 'gdp_billion_usd' in df.columns:
        last_gdp = df['gdp_billion_usd'].iloc[-1]
        avg_gdp_growth = df['gdp_billion_usd'].pct_change().mean()
        
        gdp_forecast = []
        current_year = df['year'].max()
        
        for i in range(1, forecasting_years + 6):
            future_year = current_year + i
            projected_gdp = last_gdp * ((1 + avg_gdp_growth) ** i) if avg_gdp_growth > 0 else last_gdp
            gdp_forecast.append(projected_gdp)
        
        future_gdp_values = list(df['gdp_billion_usd'].values) + gdp_forecast
        future['gdp'] = future_gdp_values[:len(future)]
    
    forecast = model.predict(future)
    
    train_end_year = df['year'].max()
    forecast = forecast[forecast['ds'].dt.year > train_end_year]
    
    actual = train_df['y'].values
    predicted = model.predict(train_df)['yhat'].values
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    metrics = {
        'model': 'Prophet',
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape': round(mape, 2),
        'training_years': len(df),
        'growth_type': growth,
        'has_gdp_regressor': add_gdp_regressor
    }
    
    return model, forecast, metrics


def generate_prophet_forecast(
    train_years: int = 2000,
    forecast_years: int = 2030,
    add_gdp_regressor: bool = True
) -> pd.DataFrame:
    """
    Generate complete Prophet forecast.
    
    Args:
        train_years: Start year for training
        forecast_years: End year for forecast
        add_gdp_regressor: Whether to use GDP regressor
    
    Returns:
        DataFrame with forecast results
    """
    from data_loader import load_demand_data
    from world_bank_data import load_world_bank_data
    from feature_engineering import engineer_features
    
    demand_df = load_demand_data(train_years, 2024)
    
    try:
        wbg_df = load_world_bank_data()
    except Exception:
        wbg_df = None
        add_gdp_regressor = False
    
    if wbg_df is not None:
        df = engineer_features(demand_df, wbg_df)
    else:
        df = demand_df.copy()
    
    forecasting_years = forecast_years - 2024
    
    model, forecast, metrics = train_prophet_model(
        df,
        add_gdp_regressor=add_gdp_regressor,
        forecasting_years=forecasting_years
    )
    
    result_df = pd.DataFrame()
    result_df['year'] = forecast['ds'].dt.year
    result_df['demand_twh'] = forecast['yhat']
    result_df['lower_ci'] = forecast['yhat_lower']
    result_df['upper_ci'] = forecast['yhat_upper']
    result_df['model'] = 'Prophet'
    
    result_df = result_df[result_df['year'] <= forecast_years]
    
    return result_df, metrics


def get_feature_importance(model: Prophet, n: int = 10) -> pd.DataFrame:
    """
    Get feature importance from Prophet model.
    
    Args:
        model: Trained Prophet model
        n: Number of top features to return
    
    Returns:
        DataFrame with feature importance
    """
    importance = model.plot_component_importance(model, n)
    
    return importance


def create_forecast_scenarios(
    df: pd.DataFrame,
    growth_scenarios: dict = None
) -> dict:
    """
    Create multiple forecast scenarios.
    
    Args:
        df: Training data
        growth_scenarios: Dict of scenario names to growth adjustments
    
    Returns:
        Dict of scenario name to forecast DataFrame
    """
    if growth_scenarios is None:
        growth_scenarios = {
            'optimistic': 1.2,
            'base': 1.0,
            'pessimistic': 0.8
        }
    
    scenarios = {}
    
    base_df = prepare_prophet_data(df)
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(base_df)
    
    future = model.make_future_dataframe(periods=6, freq='Y')
    base_forecast = model.predict(future)
    
    for scenario_name, adjustment in growth_scenarios.items():
        forecast = base_forecast.copy()
        
        demand_2024 = df['demand_twh'].iloc[-1]
        avg_growth = df['demand_twh'].pct_change().mean()
        adjusted_growth = avg_growth * adjustment
        
        for i, year in enumerate(forecast['ds'].dt.year):
            if year > 2024:
                forecast.loc[forecast['ds'].dt.year == year, 'yhat'] = (
                    demand_2024 * ((1 + adjusted_growth) ** (i + 1))
                )
        
        result = pd.DataFrame()
        result['year'] = forecast['ds'].dt.year
        result['demand_twh'] = forecast['yhat']
        result['scenario'] = scenario_name
        
        scenarios[scenario_name] = result
    
    return scenarios


if __name__ == "__main__":
    from data_loader import load_demand_data
    
    demand_df = load_demand_data(2000, 2024)
    print(f"Training on {len(demand_df)} years of data")
    
    model, forecast, metrics = train_prophet_model(
        demand_df,
        add_gdp_regressor=True,
        forecasting_years=6
    )
    
    print(f"\nModel trained successfully")
    print(f"Metrics: {metrics}")
    
    print(f"\nForecast 2025-2030:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))