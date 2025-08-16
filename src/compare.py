from pathlib import Path
import pandas as pd
from src.utils_io import ensure_dir, save_json

def summarize_metrics(csv_paths, out_json):
    rows = []
    for p in csv_paths:
        pth = Path(p)
        if pth.exists():
            df = pd.read_csv(pth)
            df['model'] = pth.stem.replace('_metrics','')
            rows.append(df)
    if not rows:
        print('No metrics found.')
        return
    cat = pd.concat(rows, ignore_index=True)
    agg = cat.groupby('model')[['MAE','RMSE','MAPE']].mean().reset_index().sort_values('RMSE')
    ensure_dir(Path(out_json).parent)
    save_json(agg.to_dict(orient='records'), out_json)
    print('Saved summary to', out_json)
