# Pakistan Energy Demand Forecasting - Project Progress

## Project Overview

This project forecasts Pakistan's electricity demand for 2025-2030 using machine learning. It extends the original data pipeline by Kareem Danish (https://github.com/karemdanish/pakistan-energy-pipeline) with forecasting capabilities.

---

## Timeline & Actions

### Session 1: Initial Setup

1. **Explored Source Repository**
   - Cloned https://github.com/karemdanish/pakistan-energy-pipeline
   - Analyzed data sources: OWID, Ember API, IRENA
   - Identified demand data: 2000-2024 (25 years)

2. **Created Project Structure**
   - Created directories: data/, models/, dashboard/, reports/
   - Set up requirements.txt
   - Created data_loader.py, world_bank_data.py, feature_engineering.py

3. **Implemented ML Models**
   - Prophet model (models/prophet_model.py)
   - ARIMA model (models/arima_model.py)

4. **Ran Pipeline**
   - Generated forecasts for 2025-2030
   - Created interactive Streamlit dashboard

5. **Pushed to GitHub**
   - Repository: https://github.com/saddamchandio/pakistan-energy-demand-forecast

### Session 2: Dashboard Fixes

**Issue 1:** Ensemble model error
- Error: KeyError: "['ensemble_lower', 'ensemble_upper'] not in index"
- Fix: Updated dashboard/app.py to properly handle Ensemble selection

**Issue 2:** Lightning icons in dashboard
- Fix: Removed all ⚡ symbols
- Updated repository links to user's GitHub

**Issue 3:** Citing original author
- Updated README.md with proper citations
- Updated sidebar to cite Kareem Danish (source data)

### Session 3: Streamlit Cloud Deployment

**Issue:** Dashboard not loading data
- Root cause: SQLite database not in repo
- Fix: 
  - Updated dashboard to load from CSV files
  - Included processed data in repo (demand_forecast.csv, merged_data.csv)

### Session 4: ARIMA Straight Line Fix

**Issue:** ARIMA showing constant values (straight line)
- Root cause: ARIMA's native prediction returns same value
- Fix: Added historical growth rate to create realistic trend
  ```python
  forecast_demand = base_forecast + (annual_growth * (i + 1))
  ```

### Session 5: Linear Trend Question

**Question:** User asked if linear trends are normal
- Answer: Yes, for short-term forecasts (6 years), linear is expected
- Explained: Historical average growth ~4.2% creates steady upward trend

### Session 6: Scenario Modeling

**Issue:** Did not account for historical negative growth years
- Problem: 5 years showed negative growth (2007, 2008, 2010, 2018, 2023)

**Solution Implemented:**
1. Added scenario modeling to ARIMA:
   - Optimistic: Uses positive growth rate (~6%)
   - Base: Uses average growth (~4.2%)
   - Pessimistic: Uses negative growth rate (~-1%)

2. Added scenario selector to dashboard

3. Updated README.md with scenario explanation

### Session 7: Comparison Chart Fixes

**Issue 1:** Single Model chart showed wrong line
- Fix: Updated create_forecast_chart() to use scenario_choice parameter

**Issue 2:** Comparison chart showed duplicate lines
- Fix: Updated create_comparison_chart() to handle scenarios

**Issue 3:** Pessimistic line going down over other lines
- Fix: When scenario selected in Comparison mode, only show that scenario's data

**Final Fix:**
- Added Prophet scenario columns (demand_prophet_optimistic, demand_prophet_pessimistic)
- Updated comparison chart to show:
  - Base: Prophet + ARIMA
  - Optimistic: Prophet (Optimistic) + ARIMA (Optimistic)
  - Pessimistic: Prophet (Pessimistic) + ARIMA (Pessimistic)

---

## Key Files Created/Modified

| File | Purpose |
|------|----------|
| requirements.txt | Python dependencies |
| data_loader.py | Load from SQLite DB |
| world_bank_data.py | Fetch GDP/population |
| feature_engineering.py | Create features |
| models/prophet_model.py | Prophet forecasting |
| models/arima_model.py | ARIMA forecasting |
| run.py | Pipeline orchestrator |
| dashboard/app.py | Streamlit dashboard |
| README.md | Project documentation |
| db/pakistan_energy.db | Original database |
| data/processed/*.csv | Processed data |

---

## Final Results

### Demand Forecast 2025-2030 (TWh)

| Year | Optimistic | Base | Pessimistic |
|------|-------------|------|-------------|
| 2025 | 196.16 | 186.82 | 183.37 |
| 2026 | 205.97 | 192.76 | 179.99 |
| 2027 | 216.26 | 198.70 | 176.67 |
| 2028 | 227.08 | 204.64 | 173.41 |
| 2029 | 238.43 | 210.58 | 170.21 |
| 2030 | 250.35 | 216.52 | 167.07 |

### Model Performance

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Prophet | 2.73 | 3.71 | 2.21% |
| ARIMA | 9.27 | 21.87 | N/A |

---

## Live Links

- **GitHub Repository:** https://github.com/saddamchandio/pakistan-energy-demand-forecast
- **Live Dashboard:** https://pakistan-energy-demand-forecast-jelw2vd7jwuj8agqgoyhqv.streamlit.app/
- **Source Data:** https://github.com/karemdanish/pakistan-energy-pipeline

---

## Author

**Saddam Hussain**

- GitHub: https://github.com/saddamchandio
- Built upon data from Kareem Danish

---

## Lessons Learned

1. **Short-term forecasts appear linear** - This is normal for 6-year horizons
2. **Always account for negative growth** - Historical data had 5 years with decline
3. **Scenario modeling is essential** - Single predictions are unreliable for planning
4. **CSV > Database for cloud deployment** - Streamlit Cloud prefers flat files

---

*Last Updated: April 16, 2026*