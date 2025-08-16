import numpy as np

def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))
def mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def evaluate_forecast(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}
