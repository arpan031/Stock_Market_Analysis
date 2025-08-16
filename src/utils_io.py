import json
from pathlib import Path
import joblib

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_pickle(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_pickle(path):
    return joblib.load(path)
