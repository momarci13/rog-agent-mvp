"""Portfolio risk & sizing math.

All formulas documented in docstrings. Tested against known cases in
tests/test_risk.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ------------------ return stats ------------------

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe: sqrt(periods) * (mean(r) - rf_per_period) / std(r)."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    rf_per = rf_annual / periods
    return float(math.sqrt(periods) * (r.mean() - rf_per) / r.std(ddof=1))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, periods: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    rf_per = rf_annual / periods
    downside = r[r < rf_per]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(math.sqrt(periods) * (r.mean() - rf_per) / downside.std(ddof=1))


def max_drawdown(equity: pd.Series) -> float:
    """MDD on an equity curve: max over t of 1 - E_t / max_{s<=t} E_s."""
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = 1 - eq / peak
    return float(dd.max())


def cagr(equity: pd.Series, periods: int = 252) -> float:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    years = len(eq) / periods
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def deflated_sharpe(
    sr_observed: float,
    returns: pd.Series,
    n_trials: int,
    sr_benchmark: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    DSR = Phi( ((SR - E[max_i SR_i]) * sqrt(T-1)) /
               sqrt(1 - gamma3*SR + ((gamma4-1)/4)*SR^2) )

    where E[max_i SR_i] is approximated by:
      (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))
    with gamma = Euler-Mascheroni ≈ 0.5772.
    """
    r = returns.dropna()
    T = len(r)
    if T < 5 or n_trials < 1:
        return 0.0

    # Moments of the returns (not annualized Sharpe)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # raw kurtosis

    # Convert annual SR to per-period for the distribution math
    sr_p = sr_observed / math.sqrt(252)
    sr_bench_p = sr_benchmark / math.sqrt(252)

    # Expected max of N iid SRs (under H0: mu=0)
    gamma_em = 0.5772156649
    N = max(2, n_trials)
    e_max = (1 - gamma_em) * stats.norm.ppf(1 - 1 / N) \
        + gamma_em * stats.norm.ppf(1 - 1 / (N * math.e))
    e_max_per = e_max / math.sqrt(T)

    denom = math.sqrt(max(1e-12, 1 - skew * sr_p + ((kurt - 1) / 4) * sr_p**2))
    z = ((sr_p - sr_bench_p - e_max_per) * math.sqrt(T - 1)) / denom
    return float(stats.norm.cdf(z))


# ------------------ sizing ------------------

def kelly_fraction(mu: float, sigma2: float, rf: float = 0.0, cap: float = 1.0) -> float:
    """Kelly: f* = (mu - rf) / sigma^2. Clip to [-cap, cap]."""
    if sigma2 <= 0:
        return 0.0
    f = (mu - rf) / sigma2
    return float(max(-cap, min(cap, f)))


def fractional_kelly(
    returns: pd.Series,
    rf_annual: float = 0.0,
    lam: float = 0.33,
    periods: int = 252,
    cap: float = 1.0,
) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    mu = r.mean() * periods
    sigma2 = r.var(ddof=1) * periods
    f_star = kelly_fraction(mu, sigma2, rf=rf_annual, cap=cap)
    return float(lam * f_star)


# ------------------ VaR ------------------

def historical_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """1-period historical VaR at confidence alpha (positive number = loss)."""
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(-np.quantile(r.values, 1 - alpha))


def parametric_var(returns: pd.Series, alpha: float = 0.99) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return float(-(mu + z * sigma))


# ------------------ MVO with L2 turnover penalty ------------------

def mvo_step(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray | None = None,
    gamma: float = 5.0,
    tau: float = 1.0,
    long_only: bool = True,
    max_weight: float = 0.2,
    max_leverage: float = 1.0,
) -> np.ndarray:
    """Solve max_w w'mu - gamma/2 w'Sigma w - tau/2 ||w - w_prev||^2
    subject to sum(w)=1, |w|<=max_weight, optional w>=0.

    Uses cvxpy if available; falls back to closed-form unconstrained solve
    if cvxpy not installed.
    """
    n = len(mu)
    w_prev = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float)

    try:
        import cvxpy as cp
        w = cp.Variable(n)
        objective = cp.Maximize(
            mu @ w
            - (gamma / 2) * cp.quad_form(w, cp.psd_wrap(sigma))
            - (tau / 2) * cp.sum_squares(w - w_prev)
        )
        cons = [cp.sum(w) == 1, cp.norm(w, 1) <= max_leverage]
        if long_only:
            cons.append(w >= 0)
        cons.append(w <= max_weight)
        cons.append(w >= -max_weight)
        prob = cp.Problem(objective, cons)
        # Try solvers in order of preference; fall back gracefully.
        for solver in ("ECOS", "SCS", "CLARABEL", "OSQP"):
            if solver in cp.installed_solvers():
                try:
                    prob.solve(solver=solver, verbose=False)
                    if w.value is not None:
                        return np.asarray(w.value).flatten()
                except Exception:
                    continue
        raise RuntimeError("No working cvxpy solver found")
    except (ImportError, RuntimeError):
        # Closed-form unconstrained: w* = (gamma*Sigma + tau*I)^-1 (mu + tau*w_prev)
        A = gamma * sigma + tau * np.eye(n)
        w = np.linalg.solve(A, mu + tau * w_prev)
        w = w / w.sum() if w.sum() != 0 else np.ones(n) / n
        if long_only:
            w = np.clip(w, 0, max_weight)
            w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
        return w


# ------------------ risk gate ------------------

@dataclass
class RiskLimits:
    max_position_pct: float = 0.20
    max_turnover: float = 0.50
    max_leverage: float = 1.0
    var_limit_pct: float = 0.05


def risk_gate(
    w_new: np.ndarray,
    w_prev: np.ndarray | None,
    returns_hist: pd.DataFrame | None,
    limits: RiskLimits,
    alpha: float = 0.99,
) -> tuple[bool, list[str]]:
    """Return (approved, reasons). Reasons list is empty on approval."""
    issues = []
    w_prev = np.zeros_like(w_new) if w_prev is None else np.asarray(w_prev)

    if np.any(np.abs(w_new) > limits.max_position_pct + 1e-9):
        issues.append(
            f"Position exceeds max {limits.max_position_pct:.0%}: "
            f"max |w| = {np.max(np.abs(w_new)):.3f}"
        )

    leverage = float(np.sum(np.abs(w_new)))
    if leverage > limits.max_leverage + 1e-9:
        issues.append(f"Leverage {leverage:.3f} > cap {limits.max_leverage}")

    turnover = float(np.sum(np.abs(w_new - w_prev)))
    if turnover > limits.max_turnover + 1e-9:
        issues.append(f"Turnover {turnover:.3f} > cap {limits.max_turnover}")

    if returns_hist is not None and not returns_hist.empty:
        port_ret = returns_hist.values @ w_new
        var = historical_var(pd.Series(port_ret), alpha=alpha)
        if var > limits.var_limit_pct + 1e-9:
            issues.append(
                f"VaR_{int(alpha*100)} = {var:.4f} > cap {limits.var_limit_pct}"
            )

    return (len(issues) == 0, issues)
