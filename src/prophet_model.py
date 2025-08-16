import pandas as pd

def fit_prophet(df: pd.DataFrame, date_col="date", value_col="close"):    # Attempt to import Prophet; return None if unavailable

    try:

        from prophet import Prophet

    except Exception as e:

        print("Prophet not available; skipping.", e)

        return None

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

    pdf = df.rename(columns={date_col:"ds", value_col:"y"})[["ds","y"]]

    m.fit(pdf)

    return m


def forecast_prophet(model, periods: int, freq="B"):

    if model is None:

        return None

    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)

    fc = model.predict(future)

    return fc[["ds","yhat","yhat_lower","yhat_upper"]]

