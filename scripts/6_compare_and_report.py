from src.config import Config
from src.compare import summarize_metrics

if __name__ == "__main__":
    cfg = Config()
    csvs = [
        cfg.reports_dir / "arima_sarima_metrics.csv",
        cfg.reports_dir / "prophet_metrics.csv",
        cfg.reports_dir / "lstm_metrics.csv",
    ]
    out = cfg.reports_dir / "summary_metrics.json"
    summarize_metrics([str(x) for x in csvs], str(out))
    print("Done.")
