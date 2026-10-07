import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

path1 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\JP Morgan.csv"
path2 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\Bank of America.csv"
path3 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\WFC.csv"
path4 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\C.csv"
path5 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\GS.csv"
path6 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\USB.csv"

df1 = pd.read_csv(path1, skiprows=[1, 2], index_col=0, parse_dates=True)
df2 = pd.read_csv(path2, skiprows=[1, 2], index_col=0, parse_dates=True)
df3 = pd.read_csv(path3, skiprows=[1, 2], index_col=0, parse_dates=True)
df4 = pd.read_csv(path4, skiprows=[1, 2], index_col=0, parse_dates=True)
df5 = pd.read_csv(path5, skiprows=[1, 2], index_col=0, parse_dates=True)
df6 = pd.read_csv(path6, skiprows=[1, 2], index_col=0, parse_dates=True)

data = pd.DataFrame({
    'JPM': df1['Close'],
    'BAC': df2['Close'],
    'WFC': df3['Close'],
    'C': df4['Close'],
    'GS': df5['Close'],
    'USB': df6['Close']
}).dropna()

#  Iterate over all unique combinations of tickers
results = []

for ticker_a, ticker_b in combinations(data.columns, 2):
    series_a = data[ticker_a]
    series_b = data[ticker_b]
    
    # Direction 1: Regress A on B (A = beta * B + const + err)
    stat_ab, pval_ab, _ = coint(series_a, series_b)
    results.append({
        'Pair': f"{ticker_a}-{ticker_b}",
        'Direction': f"{ticker_a} ~ {ticker_b}",
        'Test Statistic': stat_ab,
        'P-Value': pval_ab
    })
    
    # Direction 2: Regress B on A (B = beta * A + const + err)
    stat_ba, pval_ba, _ = coint(series_b, series_a)
    results.append({
        'Pair': f"{ticker_a}-{ticker_b}",
        'Direction': f"{ticker_b} ~ {ticker_a}",
        'Test Statistic': stat_ba,
        'P-Value': pval_ba
    })

# 2. Format results into a DataFrame sorted by lowest p-value
coint_df = pd.DataFrame(results).sort_values(by='P-Value').reset_index(drop=True)

# Display all 30 directional test runs
print("--- All Directional Cointegration Tests ---")
print(coint_df)

# 3. Filter for statistically significant cointegrated pairs (p-value < 0.05)
significant_pairs = coint_df[coint_df['P-Value'] < 0.05]

print("\n--- Statistically Significant Pairs (p < 0.05) ---")
print(significant_pairs)
