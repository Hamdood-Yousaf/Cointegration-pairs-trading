import pandas as pd
import yfinance as yf

tickers = "JPM" # Change ticker names as you like from here
start_date = "2023-01-01"
end_date = "2026-01-01"

df = yf.download(tickers=tickers, start=start_date, end=end_date)
df.to_csv("JP Morgan.csv") # Change the name in accordance to ticker name for local downloading of csv