from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def make_windows(series: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback:i+lookback+horizon])
    return np.array(X)[..., None], np.array(y)

def build_model(lookback: int, horizon: int, units: int = 64, lr: float = 1e-3) -> keras.Model:
    inp = keras.Input(shape=(lookback,1))
    x = keras.layers.LSTM(units, return_sequences=False)(inp)
    out = keras.layers.Dense(horizon)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model

def train_lstm(close: pd.Series, lookback: int=60, horizon: int=30, epochs: int=5, batch_size: int=32):
    scaler = MinMaxScaler()
    arr = close.values.reshape(-1,1)
    arr_scaled = scaler.fit_transform(arr).flatten()
    X, y = make_windows(arr_scaled, lookback, horizon)
    if len(X) < 10:
        raise ValueError("Series too short for LSTM demo; reduce lookback/horizon.")
    model = build_model(lookback, horizon)
    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(patience=2)]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)
    return model, scaler

def lstm_forecast(model, scaler, recent_values: np.ndarray, horizon: int):
    arr = recent_values.reshape(-1,1)
    arr_scaled = scaler.transform(arr).flatten()
    lookback = model.input_shape[1]
    x = arr_scaled[-lookback:]
    x = x.reshape(1, -1, 1)
    yhat_scaled = model.predict(x, verbose=0).flatten()
    yhat = scaler.inverse_transform(yhat_scaled.reshape(-1,1)).flatten()
    return yhat
