# Quantitative Trading: Core Concepts

## Momentum Anomaly

The momentum effect in equity markets, first formally documented by
Jegadeesh and Titman (1993), refers to the empirical tendency of stocks
that outperformed over the past 3 to 12 months to continue outperforming
over the next 3 to 12 months. Cross-sectionally ranking stocks on past
returns and going long the top decile while shorting the bottom decile
has historically produced economically significant alpha that survives
adjustment for the Fama-French three-factor model.

Key implementation details that affect the realized Sharpe ratio:

- **Skip month**: omit the most recent month when computing the ranking
  signal to avoid the well-known short-term reversal effect.
- **Holding period**: typical academic implementations use a 1-month
  rebalance with 6 to 12 month formation periods.
- **Liquidity filter**: restrict the universe to stocks with adequate
  average daily dollar volume (e.g., > $5M) to ensure the strategy is
  tradeable at scale and not driven by small, illiquid names.
- **Crash risk**: momentum is known to suffer severe drawdowns during
  market reversals (notably 2009 and 2020). Volatility-targeting the
  long/short portfolio mitigates but does not eliminate this risk.

## Mean-Variance Optimization

Markowitz's framework selects portfolio weights w to maximize expected
return mu^T w subject to a constraint on portfolio variance w^T Sigma w.
The unconstrained solution is

    w* = (1/gamma) * Sigma^-1 * (mu - rf * 1)

where gamma is the investor's risk aversion. In practice the inputs
mu_hat and Sigma_hat are estimated with substantial error, and naive
MVO produces unstable, extreme weights that perform poorly out of
sample. Common remedies include:

- **Shrinkage**: Ledoit-Wolf shrinkage of the sample covariance toward
  a structured target reduces estimation error.
- **Black-Litterman**: combines a market-equilibrium prior with
  investor views, producing weights closer to the market portfolio
  unless views are held with high confidence.
- **L2 turnover penalty**: adding (tau/2)||w - w_prev||^2 to the
  objective reduces transaction costs and stabilizes the rebalance.
- **Robust optimization**: replace point estimates with uncertainty
  sets and optimize the worst case within the set.

## Position Sizing via Kelly

For a strategy with estimated annualized excess return mu_hat and
variance sigma_hat^2, the Kelly-optimal capital fraction is

    f* = (mu_hat - rf) / sigma_hat^2

Full Kelly maximizes expected log-wealth but exhibits brutal drawdowns
because mu_hat is estimated with high variance. Practitioners almost
universally use fractional Kelly with lambda in [0.25, 0.5]:

    f_use = lambda * f*

A 0.25-Kelly portfolio sacrifices roughly 6% of long-run growth versus
full Kelly but cuts the volatility of returns roughly in half — usually
a desirable trade.

## Backtest Overfitting and the Deflated Sharpe Ratio

Searching over many strategy variants inflates the maximum observed
Sharpe ratio purely by chance. Bailey and López de Prado's deflated
Sharpe ratio (DSR) corrects for this by computing the probability that
the observed Sharpe exceeds what would be expected as the maximum of N
independent trials, given the empirical skew and kurtosis of the
return distribution. The recommended threshold for publishability is
DSR > 0.95.

A practical rule of thumb: if you tried 100 hyperparameter combinations
and the best one shows a Sharpe of 1.5, the deflated Sharpe is often
below 0.5 — meaning the result is plausibly noise.

## Risk Management Gates

Before any order is sent, three checks should be enforced:

1. **Position concentration**: |w_i| <= max_position_pct for all i.
2. **Leverage cap**: sum |w_i| <= max_leverage.
3. **Tail risk**: 99% VaR of the candidate portfolio, estimated via
   historical simulation on the trailing year of returns, must not
   exceed a configured fraction of equity (typically 3% to 5%).

These gates are non-negotiable regardless of what the strategy logic
proposes. They protect against bugs, regime changes, and the
occasional model that "discovered" a too-good signal because of a
look-ahead error.
