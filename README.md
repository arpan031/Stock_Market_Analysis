# Time Series Stock Market – Analysis & Forecasting

End-to-end pipeline to download stock prices, run EDA, and train + compare ARIMA/SARIMA, Prophet (optional), and LSTM models with walk‑forward backtesting and a Streamlit app.

## Quickstart
```bash
pip install -r requirements.txt
```
Then:
```bash
python scripts/0_download_data.py --ticker AAPL --start 2015-01-01 --end 2025-01-01
python scripts/1_preprocess.py --ticker AAPL
python scripts/2_train_arima_sarima.py --ticker AAPL --seasonal_m 5
python scripts/3_train_prophet.py --ticker AAPL --horizon 30
python scripts/4_train_lstm.py --ticker AAPL --epochs 5
python scripts/5_backtest_all.py --ticker AAPL --horizon 30 --folds 3
python scripts/6_compare_and_report.py
```
Streamlit:
```bash
streamlit run app/streamlit_app.py
```
