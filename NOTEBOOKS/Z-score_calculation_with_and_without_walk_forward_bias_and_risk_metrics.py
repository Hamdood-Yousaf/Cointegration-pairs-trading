import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint

path1 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\JP Morgan.csv"
path2 = r"C:\COINTEGRATION PAIRS TRADING STRATEGY\WFC.csv"

df1 = pd.read_csv(path1, skiprows=[1,2], index_col=0, parse_dates=True)
df2 = pd.read_csv(path2, skiprows=[1,2], index_col=0, parse_dates=True)

data = pd.DataFrame({
    'JPM': df1['Close'],
    'WFC': df2['Close']
}).dropna()

# ------------------------------------------------------------------
#  STATIC BETA BASELINE (LOOK-AHEAD BIAS)
# ------------------------------------------------------------------
Y = data['JPM']
X = sm.add_constant(data['WFC'])

static_model = sm.OLS(Y, X).fit()
static_beta = static_model.params['WFC']
static_alpha = static_model.params['const']

# Static Residuals & Static Z-Score
z_window = 20
data['static_residuals'] = Y - (static_alpha + static_beta * data['WFC'])
data['static_rolling_mean'] = data['static_residuals'].rolling(window=z_window).mean()
data['static_rolling_std'] = data['static_residuals'].rolling(window=z_window).std()
data['static_z_score'] = (data['static_residuals'] - data['static_rolling_mean']) / data['static_rolling_std']

# ------------------------------------------------------------------
#  WALK-FORWARD DYNAMIC BETA ESTIMATION
# ------------------------------------------------------------------
w = 252  # Rolling estimation window (~1 trading year)

# Daily re-estimation using Rolling OLS
# .shift(1) ensures beta_t is calculated strictly from data BEFORE day t
rolling_ols = RollingOLS(Y, X, window=w).fit()
data['beta_t'] = rolling_ols.params['WFC'].shift(1)
data['alpha_t'] = rolling_ols.params['const'].shift(1)

# Time-Varying Spread & Dynamic Z-Score
data['dynamic_residuals'] = Y - (data['alpha_t'] + data['beta_t'] * data['WFC'])
data['dynamic_rolling_mean'] = data['dynamic_residuals'].rolling(window=z_window).mean()
data['dynamic_rolling_std'] = data['dynamic_residuals'].rolling(window=z_window).std()
data['dynamic_z_score'] = (data['dynamic_residuals'] - data['dynamic_rolling_mean']) / data['dynamic_rolling_std']

# ------------------------------------------------------------------
#  REPORT SAMPLE LOSS (DATA BURNING) - CORRECTED
# ------------------------------------------------------------------
total_days = len(data)
usable_days = data['dynamic_z_score'].dropna().shape[0]
burned_days = total_days - usable_days  # Captures both Beta (252) + Z-score runway (19)
loss_pct = (burned_days / total_days) * 100

print("=== MILESTONE 5 SAMPLE LOSS REPORT ===")
print(f"Total Dataset Length   : {total_days} trading days")
print(f"Total Burned Days      : {burned_days} trading days (252 beta + 19 z-score runway)")
print(f"Usable Signal Days     : {usable_days} trading days")
print(f"Honest Sample Burn Ratio: {loss_pct:.2f}%")
print("======================================\n")

# ------------------------------------------------------------------
# SIGNAL GENERATION & DYNAMIC SIZING
# ------------------------------------------------------------------
signals = np.zeros(len(data))
current_position = 0

for i in range(len(data)):
    z = data['dynamic_z_score'].iloc[i]
    
    if np.isnan(z):
        signals[i] = 0
        continue

    if current_position == 0:
        if z < -2.0:
            current_position = 1    # Long spread
        elif z > 2.0:
            current_position = -1   # Short spread
    elif current_position == 1:
        if z >= 0.0:
            current_position = 0    # Exit at mean
    elif current_position == -1:
        if z <= 0.0:
            current_position = 0    # Exit at mean

    signals[i] = current_position

data['signal'] = signals

# Position Sizing utilizing time-varying Beta (beta_t)
data['position_Y_units'] = data['signal'] * 1.0
data['position_X_units'] = data['signal'] * (-data['beta_t'])

# ------------------------------------------------------------------
#  PLOT COMPARISON: WALK-FORWARD VS. STATIC Z-SCORE
# ------------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(data.index, data['static_z_score'], label='Stage 4 Static Z-Score (Look-Ahead)', color='gray', linestyle='--', alpha=0.6)
plt.plot(data.index, data['dynamic_z_score'], label='Milestone 5 Walk-Forward Z-Score', color='blue', linewidth=1.3)

plt.axhline(2.0, color='red', linestyle=':', label='Upper Threshold (+2.0)')
plt.axhline(-2.0, color='green', linestyle=':', label='Lower Threshold (-2.0)')
plt.axhline(0.0, color='black', linestyle='-', alpha=0.4)

plt.title('Walk-Forward Dynamic Z-Score vs. Static Full-Sample Z-Score (JPM ~ WFC)')
plt.xlabel('Date')
plt.ylabel('Z-Score')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Display sample of resulting output
print(data[['JPM', 'WFC', 'beta_t', 'dynamic_z_score', 'signal', 'position_Y_units', 'position_X_units']].tail(15))

# ------------------------------------------------------------------
#  PERFORMANCE METRICS & SPREAD ANALYSIS
# ------------------------------------------------------------------

# 1. Daily P&L Series
# position_t-1 is held entering day t, earning the price change (P_t - P_{t-1})
data['pnl_JPM'] = data['position_Y_units'].shift(1) * (data['JPM'] - data['JPM'].shift(1))
data['pnl_WFC'] = data['position_X_units'].shift(1) * (data['WFC'] - data['WFC'].shift(1))
data['pnl'] = data['pnl_JPM'] + data['pnl_WFC']

# Filter dataset to only include days where active trading occurs (post burn-in)
trading_data = data.dropna(subset=['dynamic_z_score', 'pnl']).copy()
trading_data['cum_pnl'] = trading_data['pnl'].cumsum()

# 2. Annualized Sharpe Ratio
daily_mean_pnl = trading_data['pnl'].mean()
daily_std_pnl = trading_data['pnl'].std()

if daily_std_pnl != 0:
    sharpe_ratio = (daily_mean_pnl / daily_std_pnl) * np.sqrt(252)
else:
    sharpe_ratio = 0.0

# 3. Max Drawdown & Date Range
cum_pnl = trading_data['cum_pnl']
running_peak = cum_pnl.cummax()
drawdown = cum_pnl - running_peak

max_drawdown = drawdown.min()
trough_date = drawdown.idxmin()
peak_date = cum_pnl.loc[:trough_date].idxmax()

# 4. Half-Life of Mean Reversion (Ornstein-Uhlenbeck fit on dynamic spread)
spread = trading_data['dynamic_residuals']
spread_lag = spread.shift(1)
spread_diff = spread - spread_lag

df_hl = pd.DataFrame({'lag': spread_lag, 'diff': spread_diff}).dropna()
X_hl = sm.add_constant(df_hl['lag'])
hl_model = sm.OLS(df_hl['diff'], X_hl).fit()

lambda_param = hl_model.params['lag']

# Half-life formula: -ln(2) / lambda
if lambda_param < 0:
    half_life = -np.log(2) / lambda_param
else:
    half_life = np.inf  # Spread is not mean-reverting if lambda >= 0

# Print Milestone 6 Summary Output
print("=== MILESTONE 6 PERFORMANCE & METRICS REPORT ===")
print(f"Annualized Sharpe Ratio : {sharpe_ratio:.4f}")
print(f"Max Drawdown ($)        : {max_drawdown:.2f}")
print(f"Max Drawdown Peak Date  : {peak_date.strftime('%Y-%m-%d')}")
print(f"Max Drawdown Trough Date: {trough_date.strftime('%Y-%m-%d')}")
if np.isinf(half_life):
    print("Estimated Half-Life     : Infinite (Spread is not mean-reverting)")
else:
    print(f"Estimated Half-Life     : {half_life:.2f} trading days")
print("=================================================\n")

# Plot Cumulative P&L and Drawdown
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(trading_data.index, trading_data['cum_pnl'], label='Cumulative Dollar P&L', color='green')
plt.axvspan(peak_date, trough_date, color='red', alpha=0.2, label='Max Drawdown Period')
plt.title('Walk-Forward Strategy Cumulative P&L')
plt.ylabel('P&L ($)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 1, 2)
plt.plot(trading_data.index, drawdown, label='Drawdown ($)', color='red')
plt.title('Strategy Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown ($)')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()