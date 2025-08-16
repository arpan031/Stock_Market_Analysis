import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(series: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,5)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_sarima(fitted, steps: int):
    fc = fitted.forecast(steps=steps)
    return np.asarray(fc)
