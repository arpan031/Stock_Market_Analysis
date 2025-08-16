import argparse
from src.config import Config
from src.preprocessing import preprocess

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cfg = Config()
    p.add_argument("--ticker", default=cfg.default_ticker)
    args = p.parse_args()
    out = preprocess(args.ticker, cfg.raw_dir, cfg.processed_dir)
    print("Processed rows:", len(out))
