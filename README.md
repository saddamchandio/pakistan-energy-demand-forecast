# Pakistan Energy Demand Forecasting

Machine learning project to forecast Pakistan's electricity demand for 2025-2030 using time series analysis.

**Building upon data from:** [Pakistan Renewable Energy Pipeline](https://github.com/karemdanish/pakistan-energy-pipeline) by [Kareem Danish](https://github.com/karemdanish)

---

## Project Overview

This project forecasts Pakistan's electricity demand using two time series models:

- **Prophet** (Meta) - Trend-based with GDP regressor
- **ARIMA** - Auto-parameter selection

The models are enhanced with World Bank economic indicators (GDP, population) and provide forecasts through 2030 with 95% confidence intervals.

---

## Data Sources

| Source | Data | Citation |
|--------|------|----------|
| Original Pipeline | Electricity demand (2000-2024) | Kareem, D. (2024) |
| World Bank | GDP, Population | World Bank (2024) |

---

## Installation

```bash
git clone https://github.com/[your-username]/pakistan-energy-demand-forecast
cd pakistan-energy-demand-forecast
pip install -r requirements.txt
```

**Note:** Ensure you have the original pipeline database (`db/pakistan_energy.db`) from the [Pakistan Renewable Energy Pipeline](https://github.com/karemdanish/pakistan-energy-pipeline).

---

## Usage

### Run the Pipeline

```bash
python run.py
```

This will:
1. Load demand data from the original database
2. Fetch World Bank GDP/population data
3. Train Prophet and ARIMA models
4. Generate forecasts for 2025-2030
5. Save results to `data/processed/`

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Historical demand visualization
- Interactive forecast charts
- Model comparison
- Downloadable forecasts

---

## Results

### Forecast 2025-2030 (TWh)

| Year | Prophet | ARIMA | Ensemble |
|------|---------|------|---------|
| 2025 | 183.52 | 181.94 | 182.73 |
| 2026 | 191.87 | 190.12 | 191.00 |
| 2027 | 200.56 | 198.45 | 199.51 |
| 2028 | 209.62 | 206.94 | 208.28 |
| 2029 | 219.06 | 215.60 | 217.33 |
| 2030 | 228.90 | 224.44 | 226.67 |

### Key Metrics

- **Latest Demand (2024):** 175.44 TWh
- **Forecast 2030:** 226.67 TWh (Ensemble)
- **CAGR (2024-2030):** 6.44%
- **Total Growth:** 29.2%

---

## Project Structure

```
pakistan-energy-demand-forecast/
├── data/
│   ├── raw/                  # World Bank data
│   └── processed/           # Merged data, forecasts
├── models/
│   ├── prophet_model.py     # Prophet implementation
│   └── arima_model.py     # ARIMA implementation
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── reports/
│   └── research_report.md # Full research report
├── data_loader.py        # Data from original DB
├── world_bank_data.py  # World Bank API
├── feature_engineering.py
├── run.py              # Pipeline runner
└── requirements.txt
```

---

## Documentation

- [Research Report](reports/research_report.md) - Detailed methodology and findings
- [Dashboard Guide](dashboard/app.py) - Interactive visualizations

---

## License

This project is for educational purposes. Data is from public sources.

---

## Citations

### Primary Data Source

```bibtex
@misc{kareem2024,
  author = {Kareem, Danish},
  title = {Pakistan Renewable Energy Pipeline},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/karemdanish/pakistan-energy-pipeline}}
}
```

### This Project

If you use this forecasting code, please cite:

```bibtex
@misc{author2025,
  author = {[Your Name]},
  title = {Pakistan Energy Demand Forecasting},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/[your-username]/pakistan-energy-demand-forecast}},
  note = {Built upon Kareem, D. (2024) Pakistan Renewable Energy Pipeline}
}
```

---

## Acknowledgments

- **Kareem Danish** - Original [Pakistan Renewable Energy Pipeline](https://github.com/karemdanish/pakistan-energy-pipeline)
- **Our World in Data** - Energy data
- **Ember** - Electricity demand data
- **IRENA** - Renewable capacity data
- **World Bank** - Economic indicators

---

## Future Work

- Add more external features (weather, sector-wise demand)
- Implement LSTM/Deep Learning models
- Regional forecasting (provincial level)
- Scenario analysis (high/low growth)

---

## Contact

For questions about this project, please refer to the project repository or contact the author.

---

*This project builds upon the excellent work of Kareem Danish in creating the Pakistan Renewable Energy Pipeline. All forecast data originates from the original pipeline.*