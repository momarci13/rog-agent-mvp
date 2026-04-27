# Risk Management: VaR, Drawdown, and Position Sizing

## 1. Introduction

Risk management is the discipline of measuring, monitoring, and controlling portfolio losses. Three critical concepts are Value at Risk (VaR), maximum drawdown (MDD), and position sizing via the Kelly criterion. This primer covers theory and practical implementation.

## 2. Value at Risk (VaR)

**Value at Risk** at confidence level $\alpha$ (e.g., $\alpha=0.99$) is the loss that will be exceeded only with probability $1-\alpha$ over a given horizon (e.g., 1 day).

### 2.1 Definition

$$\text{VaR}_\alpha = -\text{quantile}(\text{returns}, 1-\alpha)$$

For a portfolio with value $V_0$ and daily return $r$:
$$\text{VaR}_\alpha(V_0, r) = V_0 \cdot \text{VaR}_\alpha(r)$$

**Example:** If SPY has daily returns over 252 trading days, the 99% VaR is the 3rd worst day (roughly $\lceil 252 \times 0.01 \rceil = 3$).

### 2.2 Methods

- **Historical simulation** (nonparametric): Use empirical quantile. Robust, no distribution assumption needed.
- **Parametric (normal):** Assume returns $\sim N(\mu, \sigma)$, so $\text{VaR}_\alpha = \mu + \sigma \Phi^{-1}(1-\alpha)$. Simple but underestimates tail risk for fat-tailed data.
- **GARCH/EVT:** Model volatility dynamics or extreme tails separately. More sophisticated but harder to estimate.

## 3. Maximum Drawdown (MDD)

**Maximum Drawdown** is the largest peak-to-trough decline in cumulative returns:

$$\text{MDD} = \max_{t} \left(1 - \frac{\text{Equity}_t}{\max_{s \le t} \text{Equity}_s}\right)$$

Intuition: If your account peaked at $\$100k$ and fell to $\$60k$, MDD = 40%.

### 3.1 Drawdown Risk

MDD captures "disaster" risk that Sharpe ratio may miss. A strategy with:
- Sharpe = 1.5, MDD = 15% → acceptable
- Sharpe = 1.5, MDD = 60% → risky (too much capital at risk)

### 3.2 Calmar Ratio

$$\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}}$$

Balances return against worst-case drawdown. Larger is better.

## 4. Kelly Criterion for Position Sizing

The **Kelly Criterion** tells you what fraction $f$ of your capital to bet if you have a positive-expectation edge.

### 4.1 Formula

Suppose each trade has:
- Win probability: $p$
- Loss size when wrong: $L$ (as fraction of bet)
- Win size when right: $W$ (as fraction of bet)

The optimal fraction is:
$$f^* = \frac{p(1+W) - (1-p)(1+L)}{W+L} = \frac{p \cdot W - (1-p) \cdot L}{W \cdot L}$$

For a zero-mean-reversion system (e.g., momentum):
$$f^* = \frac{\mu}{\sigma^2}$$

where $\mu$ is expected return and $\sigma^2$ is variance.

### 4.2 Practical Version: Fractional Kelly

Full Kelly is often too aggressive and assumes perfect estimates. Use **fractional Kelly**:
$$f_{\text{use}} = \lambda \cdot f^*, \quad \lambda \in [0.25, 0.5]$$

**Example:** If $f^* = 0.20$ (invest 20% per trade), use $f_{\text{use}} = 0.5 \times 0.20 = 0.10$ (10%) to absorb estimation error.

## 5. Portfolio Constraints

Risk limits prevent outsized exposure:

$$\max_i |w_i| \le c_{\max}, \quad \|w\|_1 \le L_{\max}, \quad \|w - w_{\text{prev}}\|_1 \le T_{\max}, \quad \text{VaR}_\alpha(w) \le V_{\max}$$

- **Individual position cap** ($c_{\max}$): Typically 20–25% per position
- **Leverage cap** ($L_{\max}$): Sum of absolute weights ≤ 2 (50% short, 100% long, say)
- **Turnover cap** ($T_{\max}$): Max $\ell_1$ change between periods
- **VaR cap** ($V_{\max}$): Max 5% daily 99% VaR loss

## 6. Summary

Value at Risk quantifies tail risk; historical simulation is robust. Maximum drawdown captures disaster scenarios; Calmar ratio balances risk and return. Kelly Criterion optimizes leverage given an edge; always use fractional Kelly in practice. Hard constraints on positions, leverage, and turnover prevent ruin.
