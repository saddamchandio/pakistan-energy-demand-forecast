"""
Pipeline runner for Pakistan Energy Demand Forecasting.
Orchestrates data loading, feature engineering, model training, and forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


def setup_directories():
    """Create project directories if they don't exist."""
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'dashboard',
        'notebooks',
        'reports'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and merge all data sources."""
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    from data_loader import load_demand_data
    from world_bank_data import fetch_all_world_bank_data
    
    print("\n[1/3] Loading demand data from original database...")
    demand_df = load_demand_data(2000, 2024)
    print(f"  [OK] Loaded {len(demand_df)} rows of demand data")
    print(f"  [OK] Columns: {demand_df.columns.tolist()}")
    
    print("\n[2/3] Fetching World Bank data...")
    try:
        wbg_df = fetch_all_world_bank_data(2000, 2024, save=True)
        print(f"  [OK] Loaded {len(wbg_df)} rows of World Bank data")
    except Exception as e:
        print(f"  [!] World Bank API error: {e}")
        print("  [!] Proceeding without World Bank features")
        wbg_df = None
    
    print("\n[3/3] Merging datasets...")
    from feature_engineering import engineer_features
    
    if wbg_df is not None:
        merged_df = engineer_features(demand_df, wbg_df)
    else:
        merged_df = engineer_features(demand_df)
    
    print(f"  [OK] Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    output_path = Path("data/processed/merged_data.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"  [OK] Saved to: {output_path}")
    
    return demand_df, wbg_df, merged_df


def train_models(merged_df):
    """Train Prophet and ARIMA models."""
    print("\n" + "=" * 60)
    print("STEP 2: Training Models")
    print("=" * 60)
    
    from models.prophet_model import train_prophet_model
    from models.arima_model import train_arima_model
    
    results = {}
    
    print("\n[1/2] Training Prophet model...")
    try:
        prophet_model, prophet_forecast, prophet_metrics = train_prophet_model(
            merged_df,
            add_gdp_regressor=True,
            forecasting_years=6
        )
        results['Prophet'] = {
            'model': prophet_model,
            'forecast': prophet_forecast,
            'metrics': prophet_metrics
        }
        print(f"  [OK] Prophet trained: MAE={prophet_metrics['mae']}, RMSE={prophet_metrics['rmse']}")
    except Exception as e:
        print(f"  [X] Prophet error: {e}")
    
    print("\n[2/2] Training ARIMA model...")
    try:
        arima_model, arima_forecast, arima_metrics = train_arima_model(merged_df)
        results['ARIMA'] = {
            'model': arima_model,
            'forecast': arima_forecast,
            'metrics': arima_metrics
        }
        print(f"  [OK] ARIMA trained: MAE={arima_metrics['mae']}, RMSE={arima_metrics['rmse']}")
    except Exception as e:
        print(f"  [X] ARIMA error: {e}")
    
    return results


def generate_forecasts(results, demand_df):
    """Generate combined forecasts from both models."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating Forecasts")
    print("=" * 60)
    
    print("\n[1/2] Prophet forecast...")
    prophet_forecast = results.get('Prophet', {}).get('forecast')
    if prophet_forecast is not None:
        pf = prophet_forecast.copy()
        pf['year'] = pf['ds'].dt.year
        pf['demand_twh'] = pf['yhat']
        pf['lower_ci'] = pf['yhat_lower']
        pf['upper_ci'] = pf['yhat_upper']
        prophet_forecast = pf[['year', 'demand_twh', 'lower_ci', 'upper_ci']].copy()
        print(f"  [OK] Generated {len(prophet_forecast)} year forecasts")
    
    print("\n[2/2] ARIMA forecast...")
    arima_forecast = results.get('ARIMA', {}).get('forecast')
    if arima_forecast is not None:
        arima_cols = ['year', 'demand_twh', 'lower_ci', 'upper_ci']
        if 'demand_optimistic' in arima_forecast.columns:
            arima_cols.extend(['demand_optimistic', 'demand_pessimistic'])
        arima_forecast = arima_forecast[arima_cols].copy()
        print(f"  [OK] Generated {len(arima_forecast)} year forecasts")
    
    combined = pd.DataFrame()
    
    if prophet_forecast is not None:
        combined = prophet_forecast.copy()
        combined = combined.rename(columns={
            'demand_twh': 'demand_prophet',
            'lower_ci': 'prophet_lower',
            'upper_ci': 'prophet_upper'
        })
    
    if arima_forecast is not None and not combined.empty:
        arima_vals = arima_forecast.reset_index(drop=True)
        min_len = min(len(combined), len(arima_vals))
        combined = combined.iloc[:min_len].copy()
        combined = combined.reset_index(drop=True)
        combined['demand_arima'] = arima_vals['demand_twh'].iloc[:min_len].values
        combined['arima_lower'] = arima_vals['lower_ci'].iloc[:min_len].values
        combined['arima_upper'] = arima_vals['upper_ci'].iloc[:min_len].values
        
        combined['demand_ensemble'] = (
            combined['demand_prophet'] + combined['demand_arima']
        ) / 2
        
        combined['lower_ci'] = np.minimum(
            combined['prophet_lower'],
            combined['arima_lower']
        )
        combined['upper_ci'] = np.maximum(
            combined['prophet_upper'],
            combined['arima_upper']
        )
        
        if 'demand_optimistic' in arima_vals.columns:
            combined['demand_optimistic'] = arima_vals['demand_optimistic'].iloc[:min_len].values
            combined['demand_prophet_optimistic'] = (
                combined['demand_prophet'] + 
                (combined['demand_optimistic'] - combined['demand_arima'])
            )
        if 'demand_pessimistic' in arima_vals.columns:
            combined['demand_pessimistic'] = arima_vals['demand_pessimistic'].iloc[:min_len].values
            combined['demand_prophet_pessimistic'] = (
                combined['demand_prophet'] + 
                (combined['demand_pessimistic'] - combined['demand_arima'])
            )
    elif arima_forecast is not None:
        combined = arima_forecast.copy()
        combined['demand_ensemble'] = combined['demand_twh']
    
    output_path = Path("data/processed/demand_forecast.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n[OK] Saved forecasts to: {output_path}")
    
    return combined


def calculate_metrics(demand_df, results):
    """Calculate summary metrics."""
    print("\n" + "=" * 60)
    print("STEP 4: Summary Metrics")
    print("=" * 60)
    
    demand = demand_df['demand_twh']
    
    metrics = {
        'generated_at': datetime.now().isoformat(),
        'data_summary': {
            'training_years': f"{demand_df['year'].min()}-{demand_df['year'].max()}",
            'n_observations': len(demand_df),
            'latest_demand': float(demand.iloc[-1]),
            'earliest_demand': float(demand.iloc[0]),
            'avg_growth_rate': float(demand.pct_change().mean() * 100)
        },
        'model_metrics': {}
    }
    
    for model_name, model_results in results.items():
        metrics['model_metrics'][model_name] = model_results['metrics']
    
    print(f"\nTraining Period: {demand_df['year'].min()} - {demand_df['year'].max()}")
    print(f"Latest Demand (2024): {demand.iloc[-1]:.2f} TWh")
    print(f"Average Growth Rate: {demand.pct_change().mean() * 100:.2f}%")
    
    for model_name, model_results in results.items():
        m = model_results['metrics']
        print(f"\n{model_name} Model:")
        print(f"  - MAE: {m.get('mae', 'N/A')}")
        print(f"  - RMSE: {m.get('rmse', 'N/A')}")
        print(f"  - MAPE: {m.get('mape', 'N/A')}%")
    
    metrics_path = Path("data/processed/model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Saved metrics to: {metrics_path}")
    
    return metrics


def run_pipeline():
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print(" PAKISTAN ENERGY DEMAND FORECASTING PIPELINE")
    print("=" * 60)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_directories()
    
    demand_df, wbg_df, merged_df = load_data()
    
    results = train_models(merged_df)
    
    forecasts = generate_forecasts(results, demand_df)
    
    metrics = calculate_metrics(demand_df, results)
    
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved:")
    print(f"  - data/processed/merged_data.csv")
    print(f"  - data/processed/demand_forecast.csv")
    print(f"  - data/processed/model_metrics.json")
    print(f"\nNext: Run dashboard with 'streamlit run dashboard/app.py'")
    
    return results, forecasts, metrics


if __name__ == "__main__":
    results, forecasts, metrics = run_pipeline()