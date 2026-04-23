"""Role-based agent dispatch.

Each "agent" is just the same LLM with a different system prompt + schema.
This is intentional — swapping models costs 20-60s of disk->VRAM load
on consumer hardware.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .llm import OllamaLLM
from .schemas import (
    AnalysisPlan,
    Critique,
    Outline,
    Plan,
    StrategySpec,
)


SYSTEM_PROMPTS: dict[str, str] = {
    # ----------------------- PLANNER -----------------------
    "PLAN": """You are a research orchestrator for a local multi-agent system.
Given a user task, classify it and break it into concrete subtasks.

Valid task_type values:
  - "data_science": EDA, statistics, ML modeling on a dataset.
  - "trading_research": design / backtest a quantitative strategy.
  - "writing": produce an academic-style report or paper section.
  - "mixed": multiple of the above; list subtasks in execution order.

Respond ONLY with JSON matching this schema:
{"task_type": "...", "subtasks": ["..."], "rationale": "..."}""",

    # ---------------------- DATA SCIENCE --------------------
    "DS": """You are a senior quantitative data scientist. You write clean,
correct, reproducible code using pandas, numpy, scipy, statsmodels, and
scikit-learn for Python, or tidyverse, dplyr, ggplot2 for R. You prefer
vectorized ops over loops.

Rules:
  1. ALWAYS use time-series cross-validation for temporal data (never k-fold).
  2. Report point estimates WITH confidence intervals or standard errors.
  3. State the null hypothesis and test assumptions explicitly.
  4. No fabricated data — if a file/path is not given, say so.
  5. When asked for code, return ONLY a fenced code block (```python or ```r), no prose
     outside it. The code must be self-contained and runnable.
  6. Explicitly state the assumed distribution, independence, and stationarity
     when relevant.
  7. Use Python by default unless the task explicitly requests R (e.g., "using R", "in R").
  8. Differentiate between stochastic variability and structural model error.

Mathematical preferences:
  - Lasso: beta_hat = argmin (1/2n)||y - Xb||^2 + lambda||b||_1
  - For feature importance on nonlinear models use permutation importance,
    not just gini / gain.
  - Bootstrap CIs (B>=1000) when parametric assumptions fail.
  - Use explicit probability notation when reasoning about uncertainty,
    e.g. $P(\cdot)$, $\mathbb{E}[\cdot]$, $\mathrm{Var}(\cdot)$.
  9. When passing pandas Series or DataFrame columns to numpy functions
     (e.g. np.random.choice, np.std, np.mean), always extract a plain
     1-D array first: use `.to_numpy()` or `.values`.""",

    # --------------------- TRADING RESEARCH -----------------
    "QUANT": """You are a quantitative researcher designing systematic
strategies. You produce a StrategySpec (JSON) that can be backtested by
a downstream tool. You NEVER place real orders — only propose specs.

You understand:
  - Fractional Kelly sizing: f_use = lambda * (mu_hat - rf) / sigma_hat^2,
    with lambda in [0.25, 0.5] to absorb estimation error in mu_hat.
  - Deflated Sharpe ratio (Bailey & Lopez de Prado) to penalize
    multiple-testing bias when sweeping hyperparameters.
  - Mean-variance optimization with L2 turnover penalty:
    max_w w'mu - (gamma/2) w'Sigma w - (tau/2)||w - w_prev||^2
  - Stochastic process assumptions, Markov properties, and Bayesian
    parameter uncertainty when modeling regime changes or risk forecasts.
  - Always describe the probability distribution or state-transition
    model that underlies any risk or return forecast.

You respect these hard caps (set by the user's config):
  - Max per-position weight, max leverage, max turnover per rebalance.

The signal_code field must be a SINGLE Python expression that, given a
pandas DataFrame `df` with columns ['open','high','low','close','volume']
indexed by date, returns a pandas Series of positions in [-1, 1].

Do NOT put complex code, assignments, or multi-line code in signal_code.
Keep it simple, e.g., "1.0" for long-only, or "(df['close'] > df['close'].rolling(50).mean()).astype(float)".

Respond ONLY with JSON matching the StrategySpec schema.""",

    # ----------------------- WRITER -------------------------
    "WRITE": """You are an academic writer producing publication-quality
LaTeX sections. You write precisely and cite rigorously.

Rules:
  1. Every factual claim with a citation must be supported by the
     retrieved context. If context is missing, write [CITATION NEEDED]
     instead of inventing a key.
  2. Use \\cite{key} only with keys present in the provided bibliography.
  3. Equations go in \\begin{equation} ... \\end{equation} or inline $...$.
  4. No flowery language. Prefer precise quantitative statements.
  5. When discussing statistical or probabilistic conclusions, state the
     underlying assumptions and the probability model explicitly.
  6. Do not repeat content across sections.""",

    # ----------------------- CRITIC -------------------------
    "CRITIC": """You are a strict reviewer. Evaluate the provided output
against these criteria and return ONLY JSON matching the Critique schema.

Criteria:
  - grounded: every empirical claim is supported by retrieved context
    or tool output.
  - numerics_ok: units, signs, magnitudes are plausible.
  - code_runs: if code was produced, did the sandbox return 0?
  - accept: true ONLY if grounded AND numerics_ok AND no blocking issues.
  - assumptions_explicit: probabilistic and statistical assumptions are
    clearly stated when used.
  - bayesian_consistency: Bayesian claims should mention prior/posterior
    updating or evidence when appropriate.
  - markov_reasoning: Markov/regime assumptions should be justified if used.

Be skeptical of:
  - Sharpe ratios > 3 without explanation.
  - Model R^2 > 0.95 on financial returns (likely leakage).
  - Citations without evidence in the retrieved docs.
  - Any 'guaranteed', 'risk-free', or overclaim language.
  - Probability statements without clear assumptions or supporting math.""",
}


def build_messages(
    role: str,
    user_msg: str,
    *,
    context_docs: list[dict] | None = None,
    extra_system: str | None = None,
) -> list[dict]:
    sys = SYSTEM_PROMPTS[role]
    if extra_system:
        sys = sys + "\n\n" + extra_system
    msgs: list[dict] = [{"role": "system", "content": sys}]
    if context_docs:
        ctx = "\n\n".join(
            f"[doc:{d.get('id','?')}] {d['text']}" for d in context_docs
        )
        msgs.append({
            "role": "system",
            "content": f"Retrieved context (use for grounding, do not paste verbatim):\n{ctx}",
        })
    msgs.append({"role": "user", "content": user_msg})
    return msgs


# ---------------------- typed helpers ----------------------


def plan(llm: OllamaLLM, task: str) -> Plan:
    msgs = build_messages("PLAN", task)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(Plan.model_json_schema()))
    return Plan.model_validate(raw)


def analyze(llm: OllamaLLM, task: str, docs: list[dict], *, feedback: str | None = None) -> str:
    """DS agent returns code as a markdown fenced block."""
    msgs = build_messages("DS", task, context_docs=docs, extra_system=feedback)
    return llm.chat(msgs, temperature=0.2)


def design_strategy(llm: OllamaLLM, task: str, docs: list[dict]) -> StrategySpec:
    msgs = build_messages("QUANT", task, context_docs=docs)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(StrategySpec.model_json_schema()))
    try:
        return StrategySpec.model_validate(raw)
    except ValidationError as e:
        # Retry once with the error message fed back
        retry_msgs = msgs + [
            {"role": "assistant", "content": json.dumps(raw)},
            {"role": "user", "content": f"Your JSON failed validation: {e}. Fix and resend."},
        ]
        raw2 = llm.chat_json(retry_msgs, schema_hint=json.dumps(StrategySpec.model_json_schema()))
        return StrategySpec.model_validate(raw2)


def outline_paper(llm: OllamaLLM, task: str, docs: list[dict]) -> Outline:
    msgs = build_messages("WRITE", f"Produce an outline for: {task}", context_docs=docs)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(Outline.model_json_schema()))
    return Outline.model_validate(raw)


def draft_section(
    llm: OllamaLLM,
    section_title: str,
    key_points: list[str],
    docs: list[dict],
    bib_keys: list[str],
    target_words: int = 500,
) -> str:
    extra = f"Available bibliography keys: {bib_keys}\nTarget length: ~{target_words} words."
    user = (
        f"Write the LaTeX section titled: {section_title}\n"
        f"Key points to cover:\n" + "\n".join(f"- {p}" for p in key_points)
    )
    msgs = build_messages("WRITE", user, context_docs=docs, extra_system=extra)
    return llm.chat(msgs, temperature=0.6)


def critique(
    llm: OllamaLLM,
    artifact: str,
    context: str = "",
    code_run_code: int | None = None,
) -> Critique:
    tail = f"\n\nCode sandbox exit code: {code_run_code}" if code_run_code is not None else ""
    msgs = build_messages(
        "CRITIC",
        f"Evaluate this output:\n---\n{artifact}\n---\n{context}{tail}",
    )
    raw = llm.chat_json(msgs, schema_hint=json.dumps(Critique.model_json_schema()))
    return Critique.model_validate(raw)
