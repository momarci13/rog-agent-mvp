"""Typed schemas for agent I/O.

We force structured outputs because 7B models drift badly in free text.
Pydantic validation is the guardrail.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Planner ----------

TaskType = Literal["data_science", "trading_research", "writing", "mixed"]


class Plan(BaseModel):
    task_type: TaskType
    subtasks: list[str] = Field(default_factory=list)
    rationale: str = ""


# ---------- Data Science ----------

class AnalysisPlan(BaseModel):
    question: str
    data_sources: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    validation: Literal["holdout", "kfold", "timeseries_cv", "bootstrap"] = "timeseries_cv"
    metrics: list[str] = Field(default_factory=list)


# ---------- Trading ----------

class StrategySpec(BaseModel):
    name: str
    universe: list[str]                          # e.g. ["SPY","QQQ","IWM"]
    frequency: Literal["daily", "hourly"] = "daily"
    lookback_days: int = 252
    signal: str                                  # human-readable description
    signal_code: str                             # python expression over df
    position_sizing: Literal["equal_weight", "kelly", "mvo"] = "kelly"
    rebalance_days: int = 5
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


class BacktestResult(BaseModel):
    sharpe: float
    sortino: float
    deflated_sharpe: float
    max_drawdown: float
    cagr: float
    win_rate: float
    n_trades: int
    final_equity: float


# ---------- Writing ----------

class Section(BaseModel):
    title: str
    key_points: list[str] = Field(default_factory=list)
    target_words: int = 500


class Outline(BaseModel):
    title: str
    abstract: str
    sections: list[Section]


# ---------- Critic ----------

class Critique(BaseModel):
    grounded: bool | None = None
    numerics_ok: bool | None = None
    code_runs: bool | None = None
    issues: list[str] = Field(default_factory=list)
    accept: bool
    suggested_revisions: list[str] = Field(default_factory=list)
