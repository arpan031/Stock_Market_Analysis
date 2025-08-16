import matplotlib.pyplot as plt

def plot_series(df, date_col="date", value_col="close", title="Series"):
    plt.figure(figsize=(10,4))
    plt.plot(df[date_col], df[value_col])
    plt.title(title); plt.xlabel("Date"); plt.ylabel(value_col)
    plt.tight_layout()
    return plt.gcf()
