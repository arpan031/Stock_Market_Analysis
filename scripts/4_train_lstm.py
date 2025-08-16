import argparse
import pandas as pd
from src.config import Config
from src.lstm_model import train_lstm, lstm_forecast
from src.evaluate import evaluate_forecast
from src.utils_io import ensure_dir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cfg = Config()
    p.add_argument("--ticker", default=cfg.default_ticker)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    args = p.parse_args()

    df = pd.read_csv(cfg.processed_dir / f"{args.ticker}_processed.csv", parse_dates=["date"])
    close = df['close']
    model, scaler = train_lstm(close, cfg.lookback, cfg.horizon, epochs=args.epochs, batch_size=cfg.batch_size)
    recent = close.values[-cfg.lookback:]
    yhat = lstm_forecast(model, scaler, recent, cfg.horizon)

    y_true = close.values[-cfg.horizon:]
    metrics = evaluate_forecast(y_true, yhat)
    print("LSTM metrics:", metrics)

    ensure_dir(cfg.reports_dir)
    pd.DataFrame([metrics]).to_csv(cfg.reports_dir / "lstm_metrics.csv", index=False)
    pd.DataFrame({"date": df['date'].iloc[-cfg.horizon:].values, "y_true": y_true, "y_pred": yhat}).to_csv(
        cfg.reports_dir / "lstm_forecast.csv", index=False)
