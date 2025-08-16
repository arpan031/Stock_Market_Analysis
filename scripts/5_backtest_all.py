import argparse
import pandas as pd
from src.config import Config
from src.backtesting import rolling_origin_splits
from src.arima_sarima import fit_sarima, forecast_sarima
from src.evaluate import evaluate_forecast
from src.utils_io import ensure_dir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cfg = Config()
    p.add_argument("--ticker", default=cfg.default_ticker)
    p.add_argument("--folds", type=int, default=cfg.folds)
    p.add_argument("--horizon", type=int, default=cfg.horizon)
    args = p.parse_args()

    df = pd.read_csv(cfg.processed_dir / f"{args.ticker}_processed.csv", parse_dates=["date"])
    y = df['close'].values
    n = len(y)
    train_size = max(100, n - (args.folds * args.horizon) - 1)
    rows = []
    for i, (tr, te) in enumerate(rolling_origin_splits(n, train_size, args.horizon, args.folds)):
        train = y[tr]
        test = y[te]
        m = fit_sarima(pd.Series(train), order=cfg.order, seasonal_order=cfg.seasonal_order)
        yhat = forecast_sarima(m, steps=len(test))
        met = evaluate_forecast(test, yhat)
        met['fold'] = i+1
        rows.append(met)
        print(f"Fold {i+1}: {met}")
    ensure_dir(cfg.reports_dir)
    pd.DataFrame(rows).to_csv(cfg.reports_dir / "backtest_arima_sarima.csv", index=False)
