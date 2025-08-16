from pathlib import Path
import pandas as pd
from src.utils_io import ensure_dir

def preprocess(ticker: str, raw_dir: Path, processed_dir: Path):
    raw_path = Path(raw_dir) / f"{ticker}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    df = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.set_index("date").asfreq("B")
    df[["open","high","low","close","adj_close","volume"]] = df[["open","high","low","close","adj_close","volume"]].ffill()
    df["return"] = df["close"].pct_change().fillna(0.0).clip(-0.03, 0.03)
    out = df.reset_index()
    ensure_dir(processed_dir)
    out.to_csv(Path(processed_dir) / f"{ticker}_processed.csv", index=False)
    return out
