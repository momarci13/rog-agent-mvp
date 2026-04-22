# Momentum and Mean Reversion in Financial Markets

## 1. Momentum: Empirical Evidence

**Momentum** is the tendency of assets that have outperformed recently to continue outperforming in the near term.

### 1.1 Jegadeesh & Titman (1993)

Classic study on 1,241 stocks (1965–1989):
- Buy winners (top 10% by 3–12 month returns)
- Sell losers (bottom 10%)
- Hold for 3, 6, or 12 months

**Key findings:**
- Buy-and-hold returns of 12.01% per year (excess of market)
- Effect strongest at 3–12 month horizons
- Weakens at longer horizons (reversal sets in)
- Survives transaction costs and is implementable

### 1.2 Mechanism: Underreaction vs. Overreaction

Two competing theories:
- **Underreaction:** Market slowly incorporates news (behavioral); momentum strategies exploit this
- **Overreaction:** Price overshoots and reverts; profitability depends on mean reversion (statistical)

Most evidence supports underreaction at medium horizons (3–12 months) and overreaction at very long horizons (2+ years).

## 2. Mean Reversion: Theory and Evidence

**Mean Reversion** is the tendency of extreme prices to revert toward historical averages.

### 2.1 Reversal at Short Horizons

**Bid-ask bounce:** Within a day, buying then selling at midpoint incurs a round-trip cost. Thus, returns are artificially negatively autocorrelated at 1-day lag.

**Jegadeesh (1990):** Finds weak 1-month reversal effect. This is largely microstructure noise (bid-ask, intraday volatility).

### 2.2 Reversal at Long Horizons (3-5 years)

**DeBondt & Thaler (1985):**
- Buy 35 stocks with worst 5-year cumulative returns (losers)
- Sell short 35 stocks with best 5-year cumulative returns (winners)
- Hold for 5 years

**Result:** Losers outperform winners by ~25% cumulative over the next 5 years. Evidence of long-horizon overreaction.

### 2.3 Fama-French Factor Interpretation

The Fama-French 5-factor model treats momentum and reversal as separate factors:
$$r_{it} - r_{ft} = \alpha_i + \beta_i(r_{Mt} - r_{ft}) + s_i \cdot \text{SMB}_t + h_i \cdot \text{HML}_t + r_i \cdot \text{RMW}_t + c_i \cdot \text{CMA}_t + m_i \cdot \text{MOM}_t + e_{it}$$

where $\text{MOM}_t$ is the momentum factor (high-minus-low return stocks over prior 12 months).

## 3. Practical Trading Strategies

### 3.1 Cross-Sectional Momentum

**Setup:**
1. Compute 12-month returns (skipping most recent month) for all stocks in universe
2. Buy top 10% (high momentum)
3. Short bottom 10% (low momentum)
4. Rebalance monthly
5. Equal-weight or cap-weighted

**Expected Sharpe:** ~0.5–0.8 (depends on universe, costs)

### 3.2 Time-Series Momentum

**Setup:**
1. Compute 12-month return for *single* asset (e.g., SPY)
2. If return > 0: go long; else: go flat or short
3. Rebalance monthly

**Example signal code:**
```python
# 12-month return (skip last month to avoid recency)
ret_12m = (df['close'] / df['close'].shift(252)) - 1
signal = (ret_12m > 0).astype(float)
```

### 3.3 Pairs Trading (Reversal)

**Setup:**
1. Identify pairs of cointegrated assets (e.g., EWA and EWC - Australia and Canada ETFs)
2. When spread widens beyond $\mu + 2\sigma$: short spreader, long laggard
3. Exit when spread reverts to mean

**Implementation:** Use Augmented Dickey-Fuller test or Kalman filter to detect cointegration.

## 4. Why These Patterns Persist

Despite their documentation, momentum and reversal anomalies survive in modern markets:

- **Risk-based explanation:** Momentum/reversal strategies have different risk exposure than market cap; they are compensated factors, not "free money"
- **Behavioral explanation:** Investors underreact to news (momentum) and overreact to long-run information (reversal)
- **Market structure:** Transaction costs, borrowing constraints, and risk limits prevent arbitrage
- **Measurement:** True reversal is domain-specific (3–5 year for stocks; 1–2 days for futures)

## 5. Caveats and Implementation Challenges

- **Momentum crash:** During crisis periods (e.g., March 2020, long-volatility crush), momentum strategies may face rapid reversals
- **Short restrictions:** Shorting losers incurs borrow costs and may be restricted; limit to long-only or proxy via derivatives
- **Rebalancing frequency:** Monthly is standard, but daily/weekly can amplify turnover and costs
- **Universe bias:** Some universes (large-cap, liquid) show stronger momentum; small-cap shows weaker effects

## 6. Summary

Momentum (3–12 month) and long-run reversal (3–5 years) are well-documented empirical phenomena. Both survive transaction costs on large liquid universes. Implement carefully: use multiple rebalance frequencies, account for short-borrowing costs, and monitor for regime changes.
