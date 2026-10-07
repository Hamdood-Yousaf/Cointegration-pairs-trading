import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

path1 = "C:\COINTEGRATION PAIRS TRADING STRATEGY\JPM.csv"
path2 = "C:\COINTEGRATION PAIRS TRADING STRATEGY\WFC.csv"

df1 = pd.read_csv(path1, skiprows=[1, 2], index_col=0, parse_dates=True)
df2 = pd.read_csv(path2, skiprows=[1, 2], index_col=0, parse_dates=True)

# Stationarity for first differences of closing prices
df1["Difference"] = df1["Close"].diff()
difference_1 = df1["Difference"].dropna()

df2["Difference"] = df2["Close"].diff()
difference_2 = df2["Difference"].dropna()

def perform_adf_test(series, series_name="Series", alpha=0.05):

    if series.equals(df1["Close"]):
        series_name = "JPM Close prices"
    elif series.equals(df1["Difference"]):
        series_name = "JPM Differences of closing prices"
    elif series.equals(df2["Close"]):
        series_name = "WFC Close prices"
    elif series.equals(df2["Difference"]):
        series_name = "WFC Differences of closing prices"
    else:
        series_name = "Series"

    clean_series = series.dropna()
    result = adfuller(clean_series) # HERE WE ARE PASSING THE CLEAN SERIES INTO THE ADFULLER FUNCTION OF STATSMODEL
    
    adf_stat = result[0]     #THE ADF-STAT P VALUES AND CRITICAL VALUES ARE FOUND
    p_value = result[1]
    crit_values = result[4]

    print(f"\n--- ADF Test Results: {series_name} ---") # THE RESULTS FOR ADF TEST ARE PRINTED IN THESE FUNCTIONS
    print(f"ADF Statistic : {adf_stat:.6f}")
    print(f"p-value       : {p_value:.6f}")
    print("Critical Values:")
    for key, val in crit_values.items():
        print(f"   {key:>4} : {val:.4f}")
    
    if p_value <= alpha: # THIS IS USED TO MAKE FINAL CALL WETHER THE TIME-SERIES IS STATIONARY OR NON-STATIONARY
        print(f"Decision      : Reject H0 at {alpha*100:.0f}% level -> Series is STATIONARY I(0)")
    else:
        print(f"Decision      : Fail to reject H0 at {alpha*100:.0f}% level -> Series is NON-STATIONARY I(1)")

perform_adf_test(series=df1["Close"])
perform_adf_test(series=df2["Close"])
perform_adf_test(series=df1["Difference"])
perform_adf_test(series=df2["Difference"])

def plot_series(series, title="Time Series Plot", ylabel="Value", color="blue"):
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label=series.name or "Series", color=color)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(True)
    plt.legend()
    plt.show()

# Plotting WFC Close Prices
plot_series(df2["Close"], title="Wells Fargo & Company - Close Prices", ylabel="Price (USD)", color="blue")

# Plotting WFC Differences
plot_series(df2["Difference"], title="Wells Fargo & Company - First Difference", ylabel="Difference", color="black")

# Plotting JPM Close Prices
plot_series(df1["Close"], title="JP Morgan - Close Prices", ylabel="Price (USD)", color="green")

# Plotting JPM Differences
plot_series(df1["Difference"], title="JP Morgan - First Difference", ylabel="Difference", color="red")
