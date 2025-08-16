from typing import Iterator, Tuple
import numpy as np

def rolling_origin_splits(n: int, train_size: int, horizon: int, folds: int) -> Iterator[Tuple[slice, slice]]:
    step = max(1, horizon)
    for k in range(folds):
        train_end = train_size + k * step
        test_end = train_end + horizon
        if test_end > n:
            break
        yield slice(0, train_end), slice(train_end, test_end)
