# Backtesting Pitfalls and Deflated Sharpe Ratio

## 1. Introduction

Backtesting is the practice of testing a trading strategy on historical data. It is essential for understanding strategy feasibility before committing capital. However, backtesting introduces systematic biases that inflate performance metrics, particularly the Sharpe ratio. This primer covers the major pitfalls and the deflated Sharpe ratio correction.

## 2. Common Backtesting Biases

### 2.1 Look-Ahead Bias

**Look-ahead bias** occurs when future information leaks into the signal at time $t$. For example:
- Using close-of-day data to generate signals executed at open
- Using forward-filled NaN values that don't exist in real-time
- Computing indicators (e.g., rolling means) without proper alignment

**Example:** If you compute a 50-day SMA and trade at the close, you must shift the signal by 1 bar to trade on the *next* bar's open or close, not the current bar.

**Mitigation:** 
- Shift all signals by 1 bar: `signal.shift(1)`
- Explicit bar-by-bar simulation: compute indicator at $t$, execute at $t+1$

### 2.2 Survivorship Bias

Historical universes exclude delisted companies (bankruptcies, mergers). If your strategy performs well partly because losers have been removed from the dataset, performance is artificially inflated.

**Example:** Backtesting a sector rotation strategy on S&P 500 constituents. Companies like Enron (2001) or Lehman (2008) disappear from the dataset, yet they were part of the real investment opportunity set.

**Mitigation:**
- Use point-in-time index constituents (expensive data)
- Test on large liquid universes (e.g., top 500 by market cap)
- Include delisted stocks if backtesting software allows

### 2.3 Backtest Overfitting

If you optimize a strategy's parameters on historical data without proper cross-validation, you fit noise rather than signal. The fitted parameters often fail on out-of-sample data.

**Example:** Optimize SMA crossover periods (e.g., 30/200, 40/210, 50/220) on 2010–2020. The best parameters on this in-sample period may fail in 2021–2023.

**Formal Definition:** Given $N$ independent backtests (parameter combinations), the expected Sharpe ratio of the *best* one due to noise alone is approximately:

$$\mathbb{E}[\max_{i=1}^N \text{SR}_i] \approx \sqrt{\text{ln}(N)} \cdot \text{SR}(0)$$

where $\text{SR}(0) \approx \frac{1}{\sqrt{252}} \approx 0.063$ (Sharpe of random walk).

**Mitigation:**
- Out-of-sample testing: partition data into train/test
- Walk-forward optimization: slide window forward, retrain
- Purged K-fold cross-validation (López de Prado, 2018): account for temporal overlap
- Use the Deflated Sharpe Ratio (see §3)

### 2.4 Data Snooping and P-Hacking

"If you torture the data long enough, it will confess." Testing many strategies, indicators, or parameter ranges on the same historical dataset inflates false positives.

**Mitigation:**
- Pre-commit to a few strategies before backtesting
- Use Bonferroni correction or family-wise error rate control
- Report the number of trials and apply the Deflated Sharpe Ratio

## 3. Deflated Sharpe Ratio (DSR)

The Deflated Sharpe Ratio corrects for multiple testing, non-normality, and autocorrelation.

### 3.1 Formula

For an observed Sharpe ratio $\hat{\text{SR}}$ computed on $T$ returns, tested across $N$ independent trials:

$$\text{DSR} = \Phi\left(
  \frac{\big(\hat{\text{SR}} - \mathbb{E}[\max_i \text{SR}_i]\big)\sqrt{T-1}}
  {\sqrt{1 - \hat\gamma_3 \hat{\text{SR}} + \frac{\hat\gamma_4 - 1}{4}\hat{\text{SR}}^2}}
\right)$$

where:
- $\Phi$ is the standard normal CDF
- $\hat\gamma_3$ is the skewness of returns
- $\hat\gamma_4$ is the excess kurtosis (raw kurtosis minus 3)
- $\mathbb{E}[\max_i \text{SR}_i]$ is the expected maximum Sharpe under random walk

### 3.2 Expected Maximum Sharpe

$$\mathbb{E}[\max_i \text{SR}_i] \approx (1-\gamma) \Phi^{-1}\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{Ne}\right)$$

where $\gamma \approx 0.5772$ (Euler–Mascheroni constant) and $e$ is Euler's number.

For $N=1$ (no multiple testing), this simplifies to $\mathbb{E}[\max_1 \text{SR}_1] \approx 0$.

### 3.3 Interpretation

- **DSR $\approx 0.95$ or higher:** 95% confidence that the strategy's Sharpe is real (not luck)
- **DSR $\approx 0.50$:** Equal odds of real signal vs. noise
- **DSR $< 0.05$:** Likely noise; Sharpe is inflated by overfitting

**Example:** A 7B parameter-search strategy with $N=10,000$ trials, observed Sharpe=2.0 on $T=252$ days of data, high skew/kurtosis → DSR might be only 0.10–0.20, meaning the true Sharpe is likely <1.0.

## 4. Best Practices

1. **Report both Sharpe and DSR.** If DSR is low, the Sharpe is unreliable.
2. **Use out-of-sample data.** At minimum, hold out the last 20% of your data.
3. **Walk-forward rebalance.** Retrain parameters periodically (monthly/quarterly).
4. **Count your trials.** If you tested 100 parameter sets, report $N=100$ in DSR calculation.
5. **Document data cleaning.** Disclose survivorship, missing data, splits/dividends.
6. **Test on multiple universes.** If it only works on small-cap growth stocks, is it robust?

## 5. Summary

Backtesting is prone to look-ahead bias, survivorship bias, and overfitting. The Deflated Sharpe Ratio penalizes multiple testing and non-normality. A high Sharpe ratio on a small out-of-sample set is more credible than a mediocre one on in-sample data. Combine DSR with walk-forward analysis and realistic transaction costs.
