from pathlib import Path
import yfinance as yf
import pandas as pd

def download(ticker: str, start: str, end: str, interval: str, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    df.reset_index(inplace=True)
    df.rename(columns={
        "Date": "date",
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    }, inplace=True)
    df.to_csv(outdir / f"{ticker}.csv", index=False)
    return outdir / f"{ticker}.csv"
