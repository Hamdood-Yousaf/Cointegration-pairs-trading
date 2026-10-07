# Walk-Forward Cointegration & Statistical Arbitrage Pairs Trading Engine

An open source Python implementation of walk-forward statistical arbitrage strategy trading cointegrated equity pairs (JPMorgan Chase & Co. JPM vs. Wells Fargo & Co. WFC).
The system basically runs on Long term equilibrium relationship between any 2 non-stationary financial time-series, while mandating strict out of sample walk forward estimation to eliminate look ahead bias, and extract mean reverting alpha.

# Motivation

Raw stock prices behave like random walks:

$$
X_t = X_{t-1} + \epsilon_t, \quad \epsilon_t \sim \text{i.i.d.}(0, \sigma^2)
$$

Because variance grows infinitely over time fitting direct linear models across non-stationary assets leading to spurious correlations, falsely indicating strong statistical ties even though it is not truly existing

Cointegration solves this by finding a linear combination parameter or in simple words a scaling parameter ($\beta$) between two non-stationary series, so that the residual spread $\epsilon_t$ at the end is stationary (I(0)):

$$
\epsilon_t = Y_t - (\alpha + \beta X_t) \sim I(0)
$$

When two assets share an underlying economic relationship (e.g., major commercial banks JPM and WFC), their price spread acts like an elastic leash. Temporary market divergences create tradable mean reverting arbitrage opportunities

# Methodology Summary

The pipeline executes through a five stage statistical engine:

[Raw Assets: JPM & WFC] ──> [Stationarity & ADF Test] ──> [Rolling OLS (w=252)] ──> [Dynamic Z-Score Signal] ──> [Market-Neutral Execution]

1. **Asset Alignment & Cointegration Testing**:
   * Evaluates historical closing prices for JPM and WFC, verifying individual non-stationarity (I(1)) through Augmented Dickey-Fuller (ADF) testing.
   * Confirms stationarity (I(0)) of the OLS error term $\epsilon_t$.

2. **Walk-Forward Dynamic Parameter Estimation**:
   * Eliminates full-sample look-ahead bias by replacing static global regressions with a 252-trading-day ($\approx 1$ year) rolling OLS window (RollingOLS).
   * Applies a strictly causal $t-1$ lag shift (.shift(1)) so that hedge ratio $\beta_t$ and intercept $\alpha_t$ rely exclusively on past historical data rather than the data which at the time would yet not be available.

3. **Time-Varying Spread & Z-Score Normalization**:
   * Computes dynamic spread residuals: $\epsilon_t = Y_t - (\alpha_{t-1} + \beta_{t-1} X_t)$.
   * Converts raw spread deviations into a normalized Z-score using a trailing 20-day rolling window:
     $$
     Z_t = \frac{\epsilon_t - \mu_{\epsilon, 20}}{\sigma_{\epsilon, 20}}
     $$


4. **Signal Logic & Position Sizing**:
   * **Entry**: Triggers a **Long Spread** (Buy 1.0 unit $Y$, Short $\beta_t$ units $X$) when $Z_t < -2.0$, and a **Short Spread** (Short 1.0 unit $Y$, Buy $\beta_t$ units $X$) when $Z_t > +2.0$.
   * **Exit**: Closes positions when the spread reverts to equilibrium ($Z_t \to 0.0$).
   * **Market Neutrality**: Sizes the leg ratio as $1 : -\beta_t$ to insulate the portfolio against macro directional swings.

## Limitations

* **Sample Loss / Burn-In**: Dynamic estimation requires burning 271 initial trading days before generating valid, un-biased signals.
* **Frictionless Execution Assumption**: Current backtesting models assume zero trade execution costs, zero short-selling borrow fees, and continuous market liquidity.

# Results
<img width="521" height="296" alt="Screenshot 2026-10-07 193420" src="https://github.com/user-attachments/assets/7a2396e5-bc47-480b-9714-6ee2c0afa29c" />
