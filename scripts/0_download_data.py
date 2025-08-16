import argparse
from src.data_download import download
from src.config import Config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cfg = Config()
    p.add_argument("--ticker", default=cfg.default_ticker)
    p.add_argument("--start", default=cfg.start)
    p.add_argument("--end", default=cfg.end)
    p.add_argument("--interval", default=cfg.interval)
    args = p.parse_args()
    path = download(args.ticker, args.start, args.end, args.interval, cfg.raw_dir)
    print("Saved:", path)
