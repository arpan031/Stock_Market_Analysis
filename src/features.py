import pandas as pd
import numpy as np

def add_basic_features(df: pd.DataFrame):
    out = df.copy()
    out['ret'] = out['close'].pct_change().fillna(0.0)
    out['roll_mean_5'] = out['close'].rolling(5).mean().bfill()
    out['roll_std_5'] = out['close'].rolling(5).std().bfill()
    return out
