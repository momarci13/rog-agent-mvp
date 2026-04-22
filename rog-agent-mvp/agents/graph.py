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


@dataclass
class RunState:
    task: str
    task_type: str = ""
    subtasks: list[str] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    critiques: list[dict] = field(default_factory=list)
    iterations: int = 0
    accepted: bool = False


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

    # 1. PLAN
    p = roles.plan(llm, task)
    state.task_type = p.task_type
    state.subtasks = p.subtasks

    while state.iterations < max_iter and not state.accepted:
        state.iterations += 1
        docs = rag.retrieve(task, k=6)

        # 2. EXECUTE according to task type
        if state.task_type == "data_science":
            code_md = roles.analyze(llm, task, docs)
            code = _extract_code(code_md)
            run_result = {"code": "", "stdout": "", "stderr": "", "returncode": None}
            if tools.get("run_py") and code:
                run_result = tools["run_py"](code)
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

def _extract_code(md: str) -> str:
    """Pull the first ```python block out of markdown."""
    if "```python" in md:
        body = md.split("```python", 1)[1]
        return body.split("```", 1)[0].strip()
    if "```" in md:
        body = md.split("```", 1)[1]
        return body.split("```", 1)[0].strip()
    return md.strip()


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
