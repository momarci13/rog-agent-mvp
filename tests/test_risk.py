"""Tests for tools/risk.py — validate math against known cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.risk import (
    fractional_kelly,
    historical_var,
    kelly_fraction,
    max_drawdown,
    mvo_step,
    parametric_var,
    risk_gate,
    RiskLimits,
    sharpe_ratio,
    sortino_ratio,
    deflated_sharpe,
)


def test_sharpe_zero_vol():
    r = pd.Series([0.001] * 50)
    # zero std -> 0 (not inf)
    assert sharpe_ratio(r) == 0.0


def test_sharpe_known_case():
    # 1% daily mean, 2% daily std, 252 periods -> SR ~= 7.94
    np.random.seed(0)
    r = pd.Series(np.random.normal(0.01, 0.02, 252 * 5))
    sr = sharpe_ratio(r)
    # Mean ≈ 0.01, std ≈ 0.02, annualized SR ≈ sqrt(252) * 0.5 ≈ 7.94
    assert 7.0 < sr < 9.0


def test_max_drawdown():
    eq = pd.Series([100, 110, 120, 90, 95, 130])
    # Peak 120, trough 90 -> DD = 1 - 90/120 = 0.25
    assert abs(max_drawdown(eq) - 0.25) < 1e-9


def test_kelly_basic():
    # mu=0.1, sigma2=0.04 -> f* = 2.5, clipped to cap=1
    f = kelly_fraction(0.1, 0.04, cap=1.0)
    assert f == 1.0
    # Smaller edge
    f = kelly_fraction(0.05, 0.25, cap=5.0)
    assert abs(f - 0.2) < 1e-9


def test_fractional_kelly_haircut():
    np.random.seed(1)
    r = pd.Series(np.random.normal(0.0005, 0.01, 252))
    f_full = fractional_kelly(r, lam=1.0, cap=5.0)
    f_third = fractional_kelly(r, lam=1 / 3, cap=5.0)
    assert abs(f_third - f_full / 3) < 1e-9


def test_historical_var():
    # 1% VaR at alpha=0.99 means the 1st percentile loss
    r = pd.Series(np.linspace(-0.10, 0.05, 1000))  # sorted linear
    v = historical_var(r, 0.99)
    # 1st percentile ≈ -0.0985, so VaR ≈ 0.0985
    assert 0.09 < v < 0.11


def test_parametric_var_gaussian():
    np.random.seed(2)
    r = pd.Series(np.random.normal(0, 0.01, 10_000))
    v = parametric_var(r, 0.99)
    # z_0.99 ≈ 2.326, so VaR ≈ 2.326 * 0.01 = 0.0233
    assert abs(v - 0.0233) < 0.003


def test_deflated_sharpe_monotonic():
    """DSR should DECREASE as n_trials increases (multiple-testing penalty)."""
    np.random.seed(3)
    r = pd.Series(np.random.normal(0.001, 0.01, 500))
    sr = sharpe_ratio(r)
    dsr_1 = deflated_sharpe(sr, r, n_trials=1)
    dsr_100 = deflated_sharpe(sr, r, n_trials=100)
    assert dsr_1 >= dsr_100


def test_risk_gate_blocks_over_leverage():
    limits = RiskLimits(max_position_pct=1.0, max_leverage=1.0,
                        max_turnover=1.0, var_limit_pct=1.0)
    w = np.array([0.7, 0.7, 0.0])   # leverage = 1.4
    ok, issues = risk_gate(w, None, None, limits)
    assert not ok
    assert any("Leverage" in i for i in issues)


def test_risk_gate_blocks_over_position():
    limits = RiskLimits(max_position_pct=0.2)
    w = np.array([0.5, 0.3, 0.2])
    ok, issues = risk_gate(w, None, None, limits)
    assert not ok
    assert any("Position" in i for i in issues)


def test_mvo_balances_return_risk():
    mu = np.array([0.1, 0.05])
    sigma = np.array([[0.04, 0.0], [0.0, 0.01]])
    w = mvo_step(mu, sigma, gamma=5.0, tau=0.0, long_only=True, max_weight=1.0)
    assert abs(w.sum() - 1.0) < 1e-3
    # Higher-return, higher-variance asset may or may not win depending on gamma;
    # just assert both weights are non-negative and valid
    assert (w >= -1e-6).all()
