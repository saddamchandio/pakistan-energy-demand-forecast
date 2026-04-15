# Pakistan Energy Demand Forecasting

Machine learning project to forecast Pakistan's electricity demand for 2025-2030 using time series analysis.

**Live Dashboard:** https://pakistan-energy-demand-forecast-jelw2vd7jwuj8agqgoyhqv.streamlit.app/

**Built by:** Saddam Hussain   
**Repository:** https://github.com/saddamchandio/pakistan-energy-demand-forecast   
**Source Data:** https://github.com/karemdanish/pakistan-energy-pipeline

---

## Project Overview

This project forecasts Pakistan's electricity demand for 2025-2030 using two machine learning time series models. It extends the data from Kareem Danish's Pakistan Renewable Energy Pipeline by adding forecasting capabilities.

### What This Project Does

1. **Loads historical data** - Electricity demand from 2000-2024
2. **Processes data** - Creates features like lag values and growth rates
3. **Trains models** - Uses Prophet and ARIMA algorithms
4. **Generates forecasts** - Predicts demand for 2025-2030
5. **Visualizes results** - Interactive dashboard

---

## Data Source

| Source | Description | Link |
|--------|-------------|------|
| Kareem Danish | Electricity demand data (2000-2024) | [GitHub](https://github.com/karemdanish/pakistan-energy-pipeline) |

**Data includes:**
- Annual electricity demand in TWh
- Demand per capita
- Generation capacity (MW)
- Solar, wind, hydro generation data

---

## Methodology

### Models Used

**1. Prophet (Meta)**
- Decomposable time series model by Facebook/Meta
- Captures trend and yearly seasonality
- Provides 95% confidence intervals
- Best for: Data with clear trends

**2. ARIMA**
- Auto-Regressive Integrated Moving Average
- Automatic parameter selection via AIC
- Good for: Capturing autocorrelation patterns

**3. Ensemble**
- Average of Prophet and ARIMA predictions
- More balanced and robust forecast

### Growth Scenarios

This project includes **scenario modeling** to account for historical negative growth years.

| Scenario | Description | Growth Rate |
|----------|-------------|------------|
| **Optimistic** | Uses positive historical growth rates | ~6% |
| **Base** | Uses average historical growth | ~4% |
| **Pessimistic** | Accounts for negative growth years | ~-1% |

Historical data shows 5 years with negative growth:
- 2007: -3.36%
- 2008: -3.23%
- 2010: -1.61%
- 2018: -1.08%
- 2023: -2.26%

### How It Works

```
1. Load demand data from SQLite database
2. Engineer features (lag values, growth rates)
3. Train Prophet model
4. Train ARIMA model
5. Generate predictions for 2025-2030
6. Calculate ensemble average
7. Save to CSV files
8. Display in dashboard
```

---

## Installation

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
git clone https://github.com/saddamchandio/pakistan-energy-demand-forecast
cd pakistan-energy-demand-forecast
pip install -r requirements.txt
```

---

## Usage

### Run the Pipeline

```bash
python run.py
```

**What it does:**
- Loads data from database
- Trains Prophet model
- Trains ARIMA model
- Generates forecasts
- Saves to data/processed/

### Launch Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Or simply:
```bash
streamlit run dashboard/app.py
```

Then open http://localhost:8501

**Dashboard Features:**
- Historical demand chart (2000-2024)
- Forecast chart (2025-2030) with confidence intervals
- Model selection (Prophet/ARIMA/Ensemble)
- Model comparison view
- Download forecasts as CSV

---

## Results

### Forecast 2025-2030 (TWh) - With Scenarios

| Year | Optimistic | Base (ARIMA) | Pessimistic |
|------|-----------|-------------|-------------|
| 2025 | 196.16 | 186.82 | 183.37 |
| 2026 | 205.97 | 192.76 | 179.99 |
| 2027 | 216.26 | 198.70 | 176.67 |
| 2028 | 227.08 | 204.64 | 173.41 |
| 2029 | 238.43 | 210.58 | 170.21 |
| 2030 | 250.35 | 216.52 | 167.07 |

### Key Metrics

| Metric | Value |
|--------|-------|
| Latest Demand (2024) | 175.44 TWh |
| Forecast 2030 (Optimistic) | 250.35 TWh |
| Forecast 2030 (Base) | 216.52 TWh |
| Forecast 2030 (Pessimistic) | 167.07 TWh |

### Model Performance

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Prophet | 9.23 | 10.19 | 7.95% |
| ARIMA | 9.27 | 21.87 | N/A |

*MAE = Mean Absolute Error, RMSE = Root Mean Square Error, MAPE = Mean Absolute Percentage Error*

---

## Project Structure

```
pakistan-energy-demand-forecast/
├── data/
│   ├── raw/                   # Raw data files
│   └── processed/            # Processed data and forecasts
├── models/
│   ├── __init__.py
│   ├── prophet_model.py       # Prophet implementation
│   └── arima_model.py        # ARIMA implementation
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── db/
│   └── pakistan_energy.db   # SQLite database
├── run.py                   # Pipeline runner
├── setup_db.py              # Database setup
├── world_bank_data.py        # World Bank API fetching
├── feature_engineering.py     # Feature engineering
├── data_loader.py          # Data loading from DB
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## Files Explained

| File | Description |
|------|-------------|
| `run.py` | Main pipeline - coordinates everything |
| `data_loader.py` | Loads data from SQLite database |
| `world_bank_data.py` | Fetches GDP/population from World Bank |
| `feature_engineering.py` | Creates lag features and growth rates |
| `models/prophet_model.py` | Prophet model training and forecasting |
| `models/arima_model.py` | ARIMA model training and forecasting |
| `dashboard/app.py` | Interactive Streamlit dashboard |
| `db/pakistan_energy.db` | SQLite database with demand data |
| `requirements.txt` | Python dependencies |

---

## Technical Details

### Dependencies

- pandas - Data manipulation
- numpy - Numerical computing
- prophet - Time series forecasting
- statsmodels - ARIMA implementation
- streamlit - Dashboard UI
- plotly - Interactive charts
- sqlalchemy - Database access

### Data Processing

The pipeline:
1. Loads 25 years of demand data (2000-2024)
2. Creates lag features (t-1, t-2, t-3)
3. Calculates growth rates (year-over-year)
4. Computes moving averages
5. Trains both models
6. Generates 6-year forecasts

### Output Files

- `data/processed/merged_data.csv` - Combined dataset
- `data/processed/demand_forecast.csv` - Forecast results
- `data/processed/model_metrics.json` - Model performance metrics

---

## Future Improvements

Possible extensions:
- Add GDP/population as external features
- Implement LSTM deep learning model
- Add provincial/regional forecasts
- Create scenario analysis (high/low growth)
- Add weather data correlation
- Deploy to Streamlit Cloud

---

## Citation

Kareem, D. (2024). Pakistan Renewable Energy Pipeline. GitHub.

---

## License

This project is for educational purposes. Data is from public sources.

---

## Author

**Saddam Hussain**


---
