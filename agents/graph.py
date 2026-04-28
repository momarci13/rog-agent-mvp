"""Orchestrator: a tiny state loop, no LangGraph dependency.

For an MVP on a laptop we don't need a graph framework — a dict and a
while-loop does the same job in 80 lines with zero install cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.hybrid import LiteHybridRAG

from .llm import OllamaLLM
from . import roles
from .problem_decoder import decode_problem, validate_requirements
from tools import scholar


@dataclass
class RunState:
    task: str
    task_type: str = ""
    subtasks: list[str] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    critiques: list[dict] = field(default_factory=list)
    iterations: int = 0
    accepted: bool = False
    decoding: Any = None  # Problem decoding result


def run(
    task: str,
    llm: OllamaLLM,
    rag: LiteHybridRAG,
    *,
    max_iter: int = 2,
    tools: dict[str, Any] | None = None,
) -> RunState:
    """Single-shot pipeline: plan -> execute role -> critique -> revise.

    `tools` is a dict of callables the executing role may need:
      - "run_py": sandbox runner for DS role
      - "backtest": backtest engine for QUANT role
      - "latex_build": LaTeX compiler for WRITE role
    """
    tools = tools or {}
    state = RunState(task=task)

    # 0. DECODE PROBLEM
    docs_for_decoding = rag.retrieve(task, k=3)  # Quick retrieval for context
    decoding = decode_problem(llm, task, docs_for_decoding)
    validation_warnings = validate_requirements(decoding)
    if validation_warnings:
        print(f"[GRAPH] Decoding warnings: {validation_warnings}")
    state.decoding = decoding  # Add to state for later use

    # 1. PLAN
    p = roles.plan(llm, task)
    state.task_type = p.task_type
    state.subtasks = p.subtasks

    # 1.5. SCHOLAR AUGMENTATION: Search arXiv for relevant papers
    papers, scholar_context = [], ""
    try:
        papers, scholar_context = scholar.scholar_augment_task(task, n_papers=5)
        if papers:
            chunks_added = rag.ingest_papers(papers)
            print(f"[GRAPH] Ingested {len(papers)} papers ({chunks_added} chunks) into KB")
        else:
            print("[GRAPH] No relevant papers found for scholar augmentation")
    except Exception as e:
        print(f"[GRAPH] Scholar augmentation failed, continuing without it: {e}")
        scholar_context = ""

    while state.iterations < max_iter and not state.accepted:
        state.iterations += 1
        docs = rag.retrieve(task, k=6)
        
        # Add scholar context if available
        if scholar_context:
            docs.insert(0, {
                "id": "scholar_context",
                "text": scholar_context,
                "meta": {"kind": "scholar", "source": "arxiv_dynamic"}
            })

        # 2. EXECUTE according to task type
        if state.task_type == "data_science":
            feedback = None
            if state.iterations > 1 and state.artifacts:
                last_payload = state.artifacts[-1]["payload"]
                last_critique = state.critiques[-1] if state.critiques else {}
                parts = []
                stderr = last_payload.get("stderr", "").strip()
                if stderr:
                    parts.append(f"Previous run stderr:\n{stderr}")
                suggestions = last_critique.get("suggested_revisions", [])
                if suggestions:
                    parts.append("Reviewer suggestions:\n" + "\n".join(f"- {s}" for s in suggestions))
                if parts:
                    feedback = "\n\n".join(parts) + "\n\nFix these issues in your revised code."
            code_md = roles.analyze(llm, task, docs, feedback=feedback, decoding=state.decoding)
            lang, code = _extract_code(code_md)
            if code:
                if lang == "python":
                    code = _NUMPY_COMPAT + "\n" + code
                elif lang == "r":
                    code = _R_COMPAT + "\n" + code
            run_result = {"code": "", "stdout": "", "stderr": "", "returncode": None}
            if tools.get("run_py") and code and lang == "python":
                run_result = tools["run_py"](code)
                run_result["code"] = code
            elif lang == "r":
                # For R, just store the code without execution
                run_result["code"] = code
            state.artifacts.append({"type": "ds", "payload": run_result, "raw": code_md})

            # 3. CRITIQUE
            c = roles.critique(
                llm,
                f"Code:\n{code}\n\nstdout:\n{run_result.get('stdout','')[:1500]}\n"
                f"stderr:\n{run_result.get('stderr','')[:500]}",
                code_run_code=run_result.get("returncode"),
            )

        elif state.task_type == "trading_research":
            spec = roles.design_strategy(llm, task, docs)
            bt_result = None
            if tools.get("backtest"):
                bt_result = tools["backtest"](spec)
            state.artifacts.append({
                "type": "quant",
                "payload": {"spec": spec.model_dump(), "backtest": bt_result},
            })
            c = roles.critique(
                llm,
                f"Strategy: {spec.model_dump_json()}\nBacktest: {bt_result}",
            )

        elif state.task_type == "writing":
            outline = roles.outline_paper(llm, task, docs)
            sections_out = []
            bib_keys = _collect_bib_keys(rag)
            for sec in outline.sections:
                sec_docs = rag.retrieve(sec.title + " " + " ".join(sec.key_points), k=4)
                body = roles.draft_section(
                    llm, sec.title, sec.key_points, sec_docs, bib_keys, sec.target_words,
                )
                sections_out.append({"title": sec.title, "body": body})
            full_tex = _assemble_tex(outline, sections_out)
            pdf_path = None
            if tools.get("latex_build"):
                pdf_path = tools["latex_build"](full_tex)
            state.artifacts.append({
                "type": "writing",
                "payload": {"tex": full_tex, "pdf": pdf_path, "outline": outline.model_dump()},
            })
            c = roles.critique(llm, full_tex[:3000])

        elif state.task_type == "mixed":
            # naive mixed: run DS then writing
            state.task_type = "data_science"
            continue

        else:
            raise ValueError(f"Unknown task_type: {state.task_type}")

        state.critiques.append(c.model_dump())
        state.accepted = c.accept

    return state


# ---------- helpers ----------

# Patches np.random.choice to accept pandas Series/DataFrames.
# Newer numpy rejects non-array 1-D inputs; this is a common LLM codegen issue.
_NUMPY_COMPAT = """\
import numpy as _compat_np
def _compat_choice(a, *args, _orig=_compat_np.random.choice, **kwargs):
    if hasattr(a, "to_numpy"):
        a = a.to_numpy().ravel()
    return _orig(a, *args, **kwargs)
_compat_np.random.choice = _compat_choice
del _compat_np, _compat_choice

try:
    import yfinance as _compat_yf, pandas as _compat_pd
    def _compat_yf_dl(*args, _orig=_compat_yf.download, _pd=_compat_pd, **kwargs):
        df = _orig(*args, **kwargs)
        if isinstance(df.columns, _pd.MultiIndex) and df.columns.nlevels == 2:
            if df.columns.get_level_values(1).nunique() == 1:
                df.columns = df.columns.droplevel(1)
        return df
    _compat_yf.download = _compat_yf_dl
    del _compat_yf, _compat_pd, _compat_yf_dl
except ImportError:
    pass
"""

# R compatibility patches for common LLM codegen issues
_R_COMPAT = """\
# Load tidyverse if available
if (!require(tidyverse, quietly = TRUE)) {
  install.packages("tidyverse", repos = "https://cran.rstudio.com/")
  library(tidyverse)
}
# Load quantmod for financial data if available
if (!require(quantmod, quietly = TRUE)) {
  install.packages("quantmod", repos = "https://cran.rstudio.com/")
  library(quantmod)
}
"""


def _extract_code(md: str) -> tuple[str, str]:
    """Pull the first code block out of markdown, detecting language."""
    if "```r" in md:
        body = md.split("```r", 1)[1]
        return "r", body.split("```", 1)[0].strip()
    if "```python" in md:
        body = md.split("```python", 1)[1]
        return "python", body.split("```", 1)[0].strip()
    if "```" in md:
        body = md.split("```", 1)[1]
        return "unknown", body.split("```", 1)[0].strip()
    return "unknown", md.strip()


def _collect_bib_keys(rag: LiteHybridRAG) -> list[str]:
    """Return all BibTeX keys registered in the RAG store."""
    # The ingestor tags bib-derived chunks with meta={"kind":"bib","key":"..."}
    try:
        keys = []
        for m in rag.iter_metadata():
            if m.get("kind") == "bib" and m.get("key"):
                keys.append(m["key"])
        return sorted(set(keys))
    except Exception:
        return []


def _assemble_tex(outline, sections: list[dict]) -> str:
    body = "\n\n".join(f"\\section{{{s['title']}}}\n{s['body']}" for s in sections)
    return rf"""\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb,graphicx,hyperref,natbib}}
\usepackage[margin=1in]{{geometry}}
\title{{{outline.title}}}
\date{{\today}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{outline.abstract}
\end{{abstract}}

{body}

\bibliographystyle{{plainnat}}
\bibliography{{refs}}
\end{{document}}
"""
