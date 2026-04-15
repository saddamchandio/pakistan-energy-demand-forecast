"""
Streamlit Dashboard for Pakistan Energy Demand Forecasting.
Interactive visualization of historical data and forecasts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Pakistan Energy Demand Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_historical_data():
    """Load historical demand data from CSV."""
    try:
        df = pd.read_csv("data/processed/merged_data.csv")
        return df[['year', 'demand_twh']].copy()
    except Exception:
        return None


@st.cache_data
def load_forecast_data():
    """Load forecast results from CSV."""
    try:
        return pd.read_csv("data/processed/demand_forecast.csv")
    except Exception:
        return None


@st.cache_data
def load_metrics():
    """Load model metrics."""
    import json
    try:
        with open("data/processed/model_metrics.json") as f:
            return json.load(f)
    except Exception:
        return None


def get_model_forecast(model_name: str):
    """Get forecast data for specific model."""
    forecast_df = load_forecast_data()
    if forecast_df is None:
        return None, None
    
    if model_name == "Prophet":
        cols = ['year', 'demand_prophet', 'prophet_lower', 'prophet_upper']
        if all(col in forecast_df.columns for col in cols):
            return forecast_df[cols].copy(), 'Prophet'
    elif model_name == "ARIMA":
        cols = ['year', 'demand_arima', 'arima_lower', 'arima_upper']
        if all(col in forecast_df.columns for col in cols):
            return forecast_df[cols].copy(), 'ARIMA'
    elif model_name == "Ensemble":
        cols = ['year', 'demand_ensemble', 'lower_ci', 'upper_ci']
        if all(col in forecast_df.columns for col in cols):
            return forecast_df[cols].copy(), 'Ensemble'
    
    return None, model_name


def create_historical_chart(df):
    """Create historical demand chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['demand_twh'],
        mode='lines+markers',
        name='Electricity Demand',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=8, color='#00CC96')
    ))
    
    fig.update_layout(
        title={
            'text': 'Pakistan Electricity Demand (2000-2024)',
            'font': dict(size=20)
        },
        xaxis_title='Year',
        yaxis_title='Demand (TWh)',
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_forecast_chart(hist_df, forecast_df, model_name, show_ci=True):
    """Create forecast chart with confidence intervals."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['year'],
        y=hist_df['demand_twh'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=8)
    ))
    
    forecast_col = f'demand_{model_name.lower()}'
    lower_col = f'{model_name.lower()}_lower'
    upper_col = f'{model_name.lower()}_upper'
    
    if forecast_col not in forecast_df.columns:
        forecast_col = 'demand_ensemble'
        lower_col = 'lower_ci'
        upper_col = 'upper_ci'
        model_name = 'Ensemble'
    
    fig.add_trace(go.Scatter(
        x=forecast_df['year'],
        y=forecast_df[forecast_col],
        mode='lines+markers',
        name=f'{model_name} Forecast',
        line=dict(color='#FFA15A', width=3, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    if show_ci and lower_col in forecast_df.columns and upper_col in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['year'].tolist() + forecast_df['year'].tolist()[::-1],
            y=forecast_df[upper_col].tolist() + forecast_df[lower_col].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 161, 90, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title={
            'text': f'Pakistan Electricity Demand Forecast ({model_name})',
            'font': dict(size=20)
        },
        xaxis_title='Year',
        yaxis_title='Demand (TWh)',
        template='plotly_dark',
        hovermode='x unified',
        height=450
    )
    
    return fig


def create_comparison_chart(hist_df, forecast_df):
    """Create comparison chart for all models."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['year'],
        y=hist_df['demand_twh'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=8)
    ))
    
    if 'demand_prophet' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['year'],
            y=forecast_df['demand_prophet'],
            mode='lines+markers',
            name='Prophet',
            line=dict(color='#FFA15A', width=2, dash='dot'),
            marker=dict(size=6, symbol='diamond')
        ))
    
    if 'demand_arima' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['year'],
            y=forecast_df['demand_arima'],
            mode='lines+markers',
            name='ARIMA',
            line=dict(color='#19B3FF', width=2, dash='dot'),
            marker=dict(size=6, symbol='square')
        ))
    
    if 'demand_ensemble' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['year'],
            y=forecast_df['demand_ensemble'],
            mode='lines+markers',
            name='Ensemble',
            line=dict(color='#AB63FA', width=2, dash='dot'),
            marker=dict(size=6, symbol='triangle-up')
        ))
    
    fig.update_layout(
        title={
            'text': 'Model Comparison: All Forecasts',
            'font': dict(size=20)
        },
        xaxis_title='Year',
        yaxis_title='Demand (TWh)',
        template='plotly_dark',
        hovermode='x unified',
        height=450
    )
    
    return fig


def calculate_cagr(start_value, end_value, years):
    """Calculate Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return 0.0
    return ((end_value / start_value) ** (1 / years) - 1) * 100


def render_metrics_panel(hist_df, forecast_df, model_name):
    """Render key metrics panel."""
    col1, col2, col3, col4 = st.columns(4)
    
    latest_demand = hist_df['demand_twh'].iloc[-1]
    earliest_demand = hist_df['demand_twh'].iloc[0]
    years_span = hist_df['year'].max() - hist_df['year'].min()
    historical_cagr = calculate_cagr(earliest_demand, latest_demand, years_span)
    
    col1.metric(
        "Latest Demand (2024)",
        f"{latest_demand:.2f} TWh",
        delta=None
    )
    
    col2.metric(
        "Historical CAGR",
        f"{historical_cagr:.2f}%",
        help="Compound Annual Growth Rate (2000-2024)"
    )
    
    if not forecast_df.empty:
        forecast_col = f'demand_{model_name.lower()}'
        if forecast_col not in forecast_df.columns:
            forecast_col = 'demand_ensemble'
        
        if forecast_col in forecast_df.columns:
            demand_2030 = forecast_df[forecast_df['year'] == 2030][forecast_col].values
            if len(demand_2030) > 0:
                demand_2030 = demand_2030[0]
                forecast_cagr = calculate_cagr(latest_demand, demand_2030, 6)
                
                col3.metric(
                    "Forecast 2030",
                    f"{demand_2030:.2f} TWh",
                    delta=f"{forecast_cagr:.2f}% CAGR"
                )
                
                growth_rate = ((demand_2030 / latest_demand) - 1) * 100
                col4.metric(
                    "Total Growth (2024-2030)",
                    f"{growth_rate:.1f}%",
                    help="Projected growth from 2024 to 2030"
                )
    
    return {
        'latest_demand': latest_demand,
        'historical_cagr': historical_cagr
    }


def render_project_info():
    """Render project information sidebar."""
    st.sidebar.title("Pakistan Energy Forecast")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This project forecasts Pakistan's electricity demand for 2025-2030 using machine learning models (Prophet and ARIMA).
    
    **Data Source:**
    - Source data: [Kareem Danish](https://github.com/karemdanish/pakistan-energy-pipeline)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Methodology")
    st.sidebar.markdown("""
    **Models:**
    - **Prophet**: Trend-based forecasting
    - **ARIMA**: Auto-parameter selection
    - **Ensemble**: Average of both models
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Repository")
    st.sidebar.markdown("""
    - [My GitHub](https://github.com/saddamchandio/pakistan-energy-demand-forecast)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Citation")
    st.sidebar.markdown("""
    Kareem, D. (2024). Pakistan Renewable Energy Pipeline.
    """)


def main():
    """Main dashboard function."""
    render_project_info()
    
    st.title("Pakistan Electricity Demand Forecast")
    st.markdown("### Forecasting Pakistan's Energy Needs (2025-2030)")
    
    col_info, col_date = st.columns([3, 1])
    with col_date:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    hist_df = load_historical_data()
    forecast_df = load_forecast_data()
    metrics = load_metrics()
    
    if hist_df is None:
        st.error("Unable to load historical data. Please run the pipeline first: `python run.py`")
        st.stop()
    
    with st.sidebar:
        st.markdown("### Settings")
        
        model_choice = st.selectbox(
            "Select Model",
            ["Prophet", "ARIMA", "Ensemble"],
            index=0,
            help="Choose which model's forecast to display"
        )
        
        show_ci = st.checkbox(
            "Show Confidence Intervals",
            value=True,
            help="Display 95% confidence bands"
        )
        
        chart_type = st.radio(
            "Chart Type",
            ["Single Model", "Comparison"],
            help="View single model or compare all"
        )
    
    render_metrics_panel(hist_df, forecast_df if forecast_df is not None else pd.DataFrame(), model_choice)
    
    st.markdown("---")
    
    if chart_type == "Comparison" and forecast_df is not None:
        st.subheader("Model Forecasts Comparison")
        fig = create_comparison_chart(hist_df, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecast Data (2025-2030)")
        display_df = forecast_df.copy()
        
        rename_map = {
            'demand_prophet': 'Prophet',
            'demand_arima': 'ARIMA',
            'demand_ensemble': 'Ensemble'
        }
        display_df = display_df.rename(columns=rename_map)
        
        cols_to_show = ['year']
        for old, new in rename_map.items():
            if old in display_df.columns:
                cols_to_show.append(new)
        
        display_df = display_df[cols_to_show]
        st.dataframe(
            display_df.style.format({
                'Prophet': '{:.2f}',
                'ARIMA': '{:.2f}',
                'Ensemble': '{:.2f}'
            }),
            use_container_width=True
        )
        
    else:
        st.subheader(f"Demand Forecast ({model_choice})")
        
        if forecast_df is not None:
            fig = create_forecast_chart(hist_df, forecast_df, model_choice, show_ci)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Forecast Data (2025-2030)")
            
            forecast_col = f'demand_{model_choice.lower()}'
            lower_col = f'{model_choice.lower()}_lower'
            upper_col = f'{model_choice.lower()}_upper'
            
            if model_choice == "Ensemble":
                forecast_col = 'demand_ensemble'
                lower_col = 'lower_ci'
                upper_col = 'upper_ci'
            elif forecast_col not in forecast_df.columns:
                if model_choice == "Prophet":
                    forecast_col = 'demand_prophet'
                    lower_col = 'prophet_lower'
                    upper_col = 'prophet_upper'
                elif model_choice == "ARIMA":
                    forecast_col = 'demand_arima'
                    lower_col = 'arima_lower'
                    upper_col = 'arima_upper'
            
            if forecast_col in forecast_df.columns:
                display_df = forecast_df[['year', forecast_col, lower_col, upper_col]].copy()
                display_df = display_df.rename(columns={
                    forecast_col: 'Demand (TWh)',
                    lower_col: 'Lower 95% CI',
                    upper_col: 'Upper 95% CI'
                })
                
                st.dataframe(
                    display_df.style.format({
                        'Demand (TWh)': '{:.2f}',
                        'Lower 95% CI': '{:.2f}',
                        'Upper 95% CI': '{:.2f}'
                    }),
                    use_container_width=True
                )
        else:
            st.warning("No forecast data available. Run the pipeline first: `python run.py`")
    
    st.markdown("---")
    
    st.subheader("Historical Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Statistics:**")
        stats = hist_df['demand_twh'].describe()
        st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.write("**Annual Growth Rates:**")
        growth = hist_df.copy()
        growth['growth_rate'] = growth['demand_twh'].pct_change() * 100
        growth = growth[['year', 'demand_twh', 'growth_rate']].dropna()
        
        st.dataframe(
            growth.tail(10).style.format({
                'demand_twh': '{:.2f}',
                'growth_rate': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Download Forecast CSV", use_container_width=True):
            if forecast_df is not None:
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="demand_forecast.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No forecast data available")
    
    with col2:
        st.caption("""
        **Note:** This dashboard uses data from the original Pakistan Energy Pipeline 
        by Kareem Danish. Forecasts are for educational purposes only.
        """)
    
    if metrics:
        with st.expander("Model Metrics"):
            st.json(metrics)


if __name__ == "__main__":
    main()