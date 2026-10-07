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
# STATIC BETA BASELINE (LOOK-AHEAD BIAS)
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
#  REPORT SAMPLE LOSS (DATA BURNING) 
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
# 5. SIGNAL GENERATION & DYNAMIC SIZING
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
# 6. PLOT COMPARISON: WALK-FORWARD VS. STATIC Z-SCORE
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