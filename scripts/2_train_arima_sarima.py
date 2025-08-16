import argparse
from pathlib import Path
import pandas as pd
from src.config import Config
from src.arima_sarima import fit_sarima, forecast_sarima
from src.evaluate import evaluate_forecast
from src.utils_io import ensure_dir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cfg = Config()
    p.add_argument("--ticker", default=cfg.default_ticker)
    p.add_argument("--seasonal_m", type=int, default=cfg.seasonal_m)
    args = p.parse_args()

    df = pd.read_csv(cfg.processed_dir / f"{args.ticker}_processed.csv", parse_dates=["date"])
    series = df['close']
    h = cfg.horizon
    train, test = series[:-h], series[-h:]
    model = fit_sarima(train, order=cfg.order, seasonal_order=(1,1,1,args.seasonal_m))
    yhat = forecast_sarima(model, steps=h)
    metrics = evaluate_forecast(test.values, yhat)
    print("ARIMA/SARIMA metrics:", metrics)

    ensure_dir(cfg.reports_dir)
    pd.DataFrame([metrics]).to_csv(cfg.reports_dir / "arima_sarima_metrics.csv", index=False)
    pd.DataFrame({"date": df['date'].iloc[-h:].values, "y_true": test.values, "y_pred": yhat}).to_csv(
        cfg.reports_dir / "arima_sarima_forecast.csv", index=False)
