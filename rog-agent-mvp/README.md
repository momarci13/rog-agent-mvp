# ROG-Agent MVP

A laptop-grade **multi-agent RAG system** for data science, quantitative
trading research, and academic writing — running entirely on a local
free LLM via Ollama.

Designed to fit on a single consumer GPU (6–8 GB VRAM, e.g. ASUS ROG
Flow X13 with mobile RTX 4060/4070), with a clean fallback path down
to 4 GB VRAM or pure CPU.

## What it does

- **Plan → Execute → Critique** pipeline driven by a small typed state
  machine (no LangGraph dependency).
- **Three executor roles** sharing one warm model via system-prompt
  switching:
  - *Data Science* — pandas/statsmodels code generation, sandboxed
    execution, structured output.
  - *Trading Research* — produces typed `StrategySpec` JSON, runs an
    event-driven backtest, computes Sharpe / Sortino / **Deflated
    Sharpe** / MDD / VaR.
  - *Academic Writer* — outline + section drafting + LaTeX assembly,
    with citation validation against your `refs.bib`.
- **Hybrid RAG** — Chroma (dense, BGE-small embeddings) fused with BM25,
  greedy knapsack context packing under a token budget. No reranker
  (saves VRAM).
- **Risk gates** on every trading decision: position concentration,
  leverage cap, turnover cap, 99 % historical VaR.
- **Paper trading only by default.** Live execution requires an
  explicit env flag and is not wired into the agent loop.

## Architecture

```
        User task
            │
            ▼
        ┌────────┐      ┌──────────────────────────┐
        │ PLAN   │◄────►│ Hybrid RAG               │
        └────┬───┘      │  Chroma + BM25           │
             │          │  (BGE-small embeddings)  │
   ┌─────────┼──────────┴──────────────────────────┘
   ▼         ▼          ▼
┌─────┐  ┌──────┐   ┌──────┐
│ DS  │  │QUANT │   │WRITE │   ← all the same Qwen2.5-7B model
└──┬──┘  └──┬───┘   └──┬───┘     just different system prompts
   │        │          │
   ▼        ▼          ▼
sandbox  backtest   LaTeX
(stats)  (vector-   (validate
         ized py)   citations)
   │        │          │
   └────────┼──────────┘
            ▼
        ┌────────┐
        │ CRITIC │  ── revise once if not accepted
        └────────┘
```

## Why these choices for laptop hardware

- **One 7B model, role-switched.** Loading two GGUFs onto an 8 GB
  laptop GPU thrashes VRAM. We pay ~50 ms in extra prompt tokens
  instead of 30 s of swap.
- **Chroma over Qdrant.** Single-file SQLite store, no server, no
  Docker.
- **No cross-encoder reranker.** Costs ~300 MB VRAM for a marginal
  gain on a personal-scale corpus.
- **No LangGraph.** A typed `dataclass` state and a `while` loop is 80
  lines and zero install cost.
- **CPU-side stats.** `statsmodels`, `scikit-learn`, `cvxpy` all run on
  CPU, leaving the GPU for the LLM only.

## Quickstart

See **`USER_MANUAL.md`** for full setup, troubleshooting, and the
math reference. The 60-second version:

```bash
# Linux / macOS / WSL2
bash setup.sh

# Windows native
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Then:

```bash
python run.py --ingest data/papers/                                   # build KB (~30 s)
python run.py "Backtest a 50/200 SMA crossover on SPY since 2015"     # ~1-3 min
```

## Repository layout

```
rog-agent-mvp/
├── README.md                ← you are here
├── USER_MANUAL.md           ← setup, troubleshooting, math reference
├── requirements.txt
├── setup.sh / setup.ps1
├── run.py                   ← CLI entrypoint
├── configs/
│   └── config.yaml          ← model, RAG, risk knobs
├── agents/
│   ├── llm.py               ← Ollama client w/ JSON mode
│   ├── schemas.py           ← Pydantic models
│   ├── roles.py             ← 5 role prompts + typed helpers
│   └── graph.py             ← state machine
├── rag/
│   ├── hybrid.py            ← Chroma + BM25
│   └── ingest.py            ← PDF/MD/TeX/BibTeX loaders
├── tools/
│   ├── sandbox.py           ← subprocess + rlimit
│   ├── risk.py              ← Sharpe, DSR, Kelly, VaR, MVO
│   ├── backtest.py          ← event-driven backtester
│   └── tex.py               ← citation validator + LaTeX build
├── data/
│   ├── papers/              ← seed: refs.bib, quant_basics.md
│   └── market/              ← yfinance cache lands here
├── kb/                      ← Chroma persistent dir (gitignored)
├── examples/EXAMPLES.md     ← copy-paste tasks
└── tests/
    ├── test_risk.py
    ├── test_backtest.py
    └── test_rag.py
```

## What this MVP cannot do (honest list)

- Match GPT-4 / Claude planning quality. Expect to retry tasks. The
  CRITIC loop helps but doesn't close the gap.
- Long-form generation > 8k output tokens reliably. Split into sections.
- Intraday tick research — yfinance gives daily / hourly only; tick data
  is a separate problem.
- Run all roles in parallel. One model, one query at a time.
- Sustained heavy loops on battery without thermal throttling.

## Caveats

- **Not investment advice.** The Sharpe and Deflated Sharpe a backtest
  reports are estimates with substantial error. Paper-trade for months
  before risking real capital.
- **Not a security boundary.** The sandbox limits memory and CPU time
  but does not isolate the LLM from your filesystem. Don't run untrusted
  LLM-generated code on machines with sensitive data.
- **Citations need human review.** The validator strips invented BibTeX
  keys, but it can't tell whether the cited claim is correctly
  represented. Read the output before publishing anything.

## License

MIT — do whatever you want, no warranty.
