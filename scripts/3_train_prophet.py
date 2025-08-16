import argparse
import pandas as pd
from src.config import Config
from src.prophet_model import fit_prophet, forecast_prophet
from src.evaluate import evaluate_forecast
from src.utils_io import ensure_dir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cfg = Config()
    p.add_argument("--ticker", default=cfg.default_ticker)
    p.add_argument("--horizon", type=int, default=cfg.horizon)
    args = p.parse_args()

    df = pd.read_csv(cfg.processed_dir / f"{args.ticker}_processed.csv", parse_dates=["date"])
    model = fit_prophet(df, "date", "close")
    fc = forecast_prophet(model, periods=args.horizon, freq="B")
    if fc is None:
        print("Prophet not available; skipping.")
    else:
        y_true = df['close'].iloc[-args.horizon:].values
        y_pred = fc['yhat'].values[:args.horizon]
        metrics = evaluate_forecast(y_true, y_pred)
        print("Prophet metrics:", metrics)
        ensure_dir(cfg.reports_dir)
        pd.DataFrame([metrics]).to_csv(cfg.reports_dir / "prophet_metrics.csv", index=False)
        pd.DataFrame({"date": df['date'].iloc[-args.horizon:].values, "y_true": y_true, "y_pred": y_pred}).to_csv(
            cfg.reports_dir / "prophet_forecast.csv", index=False)
