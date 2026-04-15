"""
ARIMA model implementation for Pakistan Energy Demand Forecasting.
Uses statsmodels for ARIMA time series forecasting.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(series: pd.Series) -> dict:
    """
    Check stationarity of a time series using ADF test.
    
    Args:
        series: Time series data
    
    Returns:
        Dict with test results
    """
    result = adfuller(series.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def get_arima_parameters(series: pd.Series) -> dict:
    """
    Automatically determine ARIMA parameters using AIC.
    
    Args:
        series: Time series data
    
    Returns:
        Dict with best (p, d, q) parameters
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    
    d = 1
    
    series_diff = series
    stationarity = check_stationarity(series)
    if stationarity['is_stationary']:
        d = 0
    
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    
            except Exception:
                continue
    
    return {
        'order': best_order,
        'p': best_order[0],
        'd': best_order[1],
        'q': best_order[2],
        'aic': best_aic
    }


def train_arima_model(
    df: pd.DataFrame,
    order: Optional[Tuple[int, int, int]] = None,
    use_auto: bool = True
) -> Tuple[ARIMA, pd.DataFrame, dict]:
    """
    Train ARIMA model on demand data.
    
    Args:
        df: DataFrame with 'year' and 'demand_twh' columns
        order: ARIMA order (p, d, q). If None, auto-determine
        use_auto: Whether to use automatic parameter selection
    
    Returns:
        Tuple of (trained_model, forecast_df, metrics)
    """
    series = df.set_index('year')['demand_twh']
    
    if use_auto or order is None:
        params = get_arima_parameters(series)
        order = params['order']
    
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    train_end = len(series)
    forecast_periods = 6
    
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    
    future_years = list(range(series.index.max() + 1, series.index.max() + forecast_periods + 1))
    
    forecast_df = pd.DataFrame()
    forecast_df['year'] = future_years
    forecast_df['demand_twh'] = forecast_mean.values
    forecast_df['lower_ci'] = conf_int.iloc[:, 0].values
    forecast_df['upper_ci'] = conf_int.iloc[:, 1].values
    forecast_df['model'] = 'ARIMA'
    
    residuals = model_fit.resid
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    aic = model_fit.aic
    bic = model_fit.bic
    
    metrics = {
        'model': 'ARIMA',
        'order': order,
        'p': order[0],
        'd': order[1],
        'q': order[2],
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'aic': round(aic, 2),
        'bic': round(bic, 2),
        'training_years': len(df)
    }
    
    return model_fit, forecast_df, metrics


def generate_arima_forecast(
    train_years: int = 2000,
    forecast_years: int = 2030,
    order: Optional[Tuple[int, int, int]] = None
) -> pd.DataFrame:
    """
    Generate complete ARIMA forecast.
    
    Args:
        train_years: Start year for training
        forecast_years: End year for forecast
        order: ARIMA order (optional)
    
    Returns:
        DataFrame with forecast results
    """
    from data_loader import load_demand_data
    
    demand_df = load_demand_data(train_years, 2024)
    
    forecasting_years = forecast_years - 2024
    
    model_fit, forecast, metrics = train_arima_model(demand_df, order=order)
    
    result_df = forecast.copy()
    result_df = result_df[result_df['year'] <= forecast_years]
    
    return result_df, metrics


def create_forecast_scenarios(
    df: pd.DataFrame,
    scenarios: dict = None
) -> dict:
    """
    Create multiple forecast scenarios using different orders.
    
    Args:
        df: Training data
        scenarios: Dict of scenario names to (p, d, q) orders
    
    Returns:
        Dict of scenario name to forecast DataFrame
    """
    if scenarios is None:
        scenarios = {
            'conservative': (1, 1, 1),
            'moderate': (2, 1, 2),
            'aggressive': (3, 1, 3)
        }
    
    results = {}
    
    for scenario_name, order in scenarios.items():
        try:
            model_fit, forecast, metrics = train_arima_model(df, order=order)
            forecast['scenario'] = scenario_name
            results[scenario_name] = forecast
        except Exception as e:
            print(f"Failed for {scenario_name} with order {order}: {e}")
    
    return results


def get_residual_analysis(model_fit: ARIMA) -> dict:
    """
    Get residual analysis from fitted model.
    
    Args:
        model_fit: Fitted ARIMA model
    
    Returns:
        Dict with residual statistics
    """
    residuals = model_fit.resid
    
    return {
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'min': float(residuals.min()),
        'max': float(residuals.max()),
        'skew': float(residuals.skew()),
        'kurtosis': float(residuals.kurtosis())
    }


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare different ARIMA orders.
    
    Args:
        df: Training data
    
    Returns:
        DataFrame with comparison metrics
    """
    orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 1),
        (1, 1, 3)
    ]
    
    results = []
    
    series = df.set_index('year')['demand_twh']
    
    for order in orders:
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            
            results.append({
                'order': order,
                'AIC': model_fit.aic,
                'BIC': model_fit.bic,
                'MAE': np.mean(np.abs(model_fit.resid)),
                'RMSE': np.sqrt(np.mean(model_fit.resid ** 2))
            })
        except Exception:
            continue
    
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values('AIC')
    
    return comparison


if __name__ == "__main__":
    from data_loader import load_demand_data
    
    demand_df = load_demand_data(2000, 2024)
    print(f"Training on {len(demand_df)} years of data")
    
    params = get_arima_parameters(demand_df.set_index('year')['demand_twh'])
    print(f"Auto-selected parameters: {params}")
    
    model, forecast, metrics = train_arima_model(demand_df, order=params['order'])
    
    print(f"\nModel trained successfully")
    print(f"Metrics: {metrics}")
    
    print(f"\nForecast 2025-2030:")
    print(forecast)
    
    comparison = compare_models(demand_df)
    print(f"\nModel Comparison:")
    print(comparison)