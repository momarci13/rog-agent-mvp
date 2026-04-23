from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from run import load_config, make_tools, run_kan_demo
from agents.llm import LLMConfig, OllamaLLM
from tools.backtest import BacktestConfig, compile_signal, fetch_yahoo, run_portfolio_backtest

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT_RUNS = ROOT / "output" / "runs"
OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ROG-Agent Local UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


def _config() -> dict[str, Any]:
    return load_config(str(CONFIG_PATH))


def _llm(cfg: dict[str, Any]) -> OllamaLLM:
    llm_cfg = LLMConfig(
        model=cfg["llm"]["model"],
        host=cfg["llm"]["host"],
        num_ctx=cfg["llm"]["num_ctx"],
        temperature=cfg["llm"]["temperature"],
        timeout_s=cfg["llm"]["timeout_s"],
    )
    return OllamaLLM(llm_cfg)


def _rag(cfg: dict[str, Any]):
    try:
        from rag.hybrid import LiteHybridRAG
    except Exception as exc:
        raise HTTPException(500, f"RAG backend unavailable: {exc}")
    return LiteHybridRAG(
        db_path=cfg["rag"]["db_path"],
        embedding_model=cfg["rag"]["embedding_model"],
        alpha_dense=cfg["rag"]["alpha_dense"],
    )


def _save_run(state: Any) -> Path:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    idx = len(list(OUTPUT_RUNS.glob("run_*.json")))
    run_path = OUTPUT_RUNS / f"run_{idx:04d}.json"
    run_path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")
    return run_path


class TaskRequest(BaseModel):
    task: str


class IngestRequest(BaseModel):
    path: str


@app.get("/", response_class=FileResponse)
async def index():
    return ROOT / "web" / "index.html"


@app.get("/health")
async def health() -> dict[str, Any]:
    cfg = _config()
    llm = _llm(cfg)
    llm_ok = False
    try:
        llm_ok = llm.health()
    except Exception as exc:
        return {"status": "error", "detail": f"LLM health check failed: {exc}"}

    try:
        rag = _rag(cfg)
        rag_count = len(rag)
    except HTTPException as exc:
        rag_count = None
        return {"status": "partial", "llm": llm_ok, "rag": str(exc.detail)}

    return {"status": "ok", "llm": llm_ok, "rag_chunks": rag_count}


@app.post("/run-task")
async def run_task(payload: TaskRequest) -> dict[str, Any]:
    cfg = _config()
    llm = _llm(cfg)
    if not llm.health():
        raise HTTPException(500, "LLM is not healthy or unreachable.")

    rag = _rag(cfg)
    tools = make_tools(cfg)
    try:
        from agents.graph import run
    except Exception as exc:
        raise HTTPException(500, f"Agents graph unavailable: {exc}")

    state = run(payload.task, llm, rag, max_iter=1, tools=tools)
    run_path = _save_run(state)
    return {"status": "ok", "path": str(run_path), "run": asdict(state)}


@app.post("/ingest")
async def ingest(payload: IngestRequest) -> dict[str, Any]:
    cfg = _config()
    try:
        from rag.hybrid import LiteHybridRAG
        from rag.ingest import ingest_path
    except Exception as exc:
        raise HTTPException(500, f"RAG ingestion unavailable: {exc}")

    rag = LiteHybridRAG(
        db_path=cfg["rag"]["db_path"],
        embedding_model=cfg["rag"]["embedding_model"],
        alpha_dense=cfg["rag"]["alpha_dense"],
    )
    n = ingest_path(payload.path, rag, chunk_tokens=cfg["rag"]["chunk_tokens"])
    return {"status": "ok", "chunks": len(rag), "added": n}


@app.get("/kan-demo")
async def kan_demo() -> dict[str, Any]:
    result = run_kan_demo()
    return {"status": "ok", "demo": result}


@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUT_RUNS.glob("run_*.json"))
    return {"runs": [f.name for f in files]}


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run_path = OUTPUT_RUNS / run_id
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(404, "Run file not found")
    content = run_path.read_text(encoding="utf-8")
    return json.loads(content)
