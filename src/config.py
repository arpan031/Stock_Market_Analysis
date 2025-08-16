from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    data_dir: Path = Path("data")
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    default_ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    interval: str = "1d"
    test_size: int = 60
    val_size: int = 60
    horizon: int = 30
    folds: int = 3
    random_state: int = 42
    lookback: int = 60
    lstm_units: int = 64
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    seasonal_m: int = 5
    order: tuple = (1,1,1)
    seasonal_order: tuple = (1,1,1,5)
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
