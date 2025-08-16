import pandas as pd
import streamlit as st
from src.config import Config
from src.arima_sarima import fit_sarima, forecast_sarima

st.set_page_config(page_title="Stock Forecasting", layout="wide")
cfg = Config()

st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Ticker", cfg.default_ticker)
horizon = st.sidebar.number_input("Horizon (days)", value=cfg.horizon, min_value=5, max_value=120, step=5)

data_file = cfg.processed_dir / f"{ticker}_processed.csv"
if not data_file.exists():
    st.warning(f"Processed file not found: {data_file}. Run preprocessing scripts first.")
else:
    df = pd.read_csv(data_file, parse_dates=["date"]).sort_values("date")
    st.subheader(f"{ticker} Close Price")
    st.line_chart(df.set_index("date")["close"])

    series = df["close"]
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    model = fit_sarima(train)
    yhat = forecast_sarima(model, steps=horizon)

    st.subheader("Forecast vs Actuals")
    chart_df = pd.DataFrame({
        "date": df["date"].iloc[-horizon:],
        "actual": test.values,
        "forecast": yhat
    }).set_index("date")
    st.line_chart(chart_df)
