from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from run import load_config, make_llm_config, make_tools, run_kan_demo
from agents.llm import LLMConfig, OllamaLLM
from tools.backtest import BacktestConfig, compile_signal, run_portfolio_backtest
from tools.data import fetch_yahoo
from tools import task_storage, task_conversation

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


# Startup: run migration from legacy format
@app.on_event("startup")
async def startup_migration():
    try:
        migration_result = task_storage.migrate_legacy_runs()
        if migration_result["migrated"] > 0:
            print(f"[STARTUP] Migrated {migration_result['migrated']} legacy tasks")
    except Exception as e:
        print(f"[STARTUP] Migration warning: {e}")


def _config() -> dict[str, Any]:
    return load_config(str(CONFIG_PATH))


def _llm(cfg: dict[str, Any]) -> OllamaLLM:
    llm_cfg = make_llm_config(cfg)
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
        query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
        query_expansion_method=cfg["rag"]["query_expansion"]["method"],
        max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
        reranking_enabled=cfg["rag"]["reranking"]["enabled"],
        reranking_model=cfg["rag"]["reranking"]["model"],
        top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
        top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
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


class MessageRequest(BaseModel):
    content: str
    iteration: int = 0


class BranchRequest(BaseModel):
    branch_name: str | None = None
    from_iteration: int = 0


class ReRunRequest(BaseModel):
    code: str


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
    task_id = task_storage.save_task(state)
    return {"status": "ok", "task_id": task_id, "run": task_storage._serialize_for_json(state)}


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
        query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
        query_expansion_method=cfg["rag"]["query_expansion"]["method"],
        max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
        reranking_enabled=cfg["rag"]["reranking"]["enabled"],
        reranking_model=cfg["rag"]["reranking"]["model"],
        top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
        top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
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


# New Task Management API Endpoints
@app.get("/api/tasks")
async def list_tasks_api(limit: int = 50, offset: int = 0, sort_by: str = "-updated_at") -> dict[str, Any]:
    """List all tasks with pagination and sorting."""
    try:
        tasks, total = task_storage.list_tasks(limit=limit, offset=offset, sort_by=sort_by)
        return {"tasks": tasks, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(500, f"Failed to list tasks: {str(e)}")


@app.get("/api/tasks/{task_id}")
async def get_task_api(task_id: str) -> dict[str, Any]:
    """Get full task with all messages and artifacts."""
    try:
        task = task_storage.load_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        
        # Serialize task for JSON response
        task_data = task_storage._serialize_for_json(task)
        return {"task": task_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load task: {str(e)}")


@app.get("/api/tasks/search")
async def search_tasks_api(q: str, limit: int = 20) -> dict[str, Any]:
    """Search tasks by keyword."""
    try:
        results = task_storage.search_tasks(q, limit=limit)
        return {"results": results, "query": q, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.post("/api/tasks/{task_id}/messages")
async def add_task_message(task_id: str, payload: MessageRequest) -> dict[str, Any]:
    """Add a user message and get assistant response."""
    try:
        cfg = _config()
        llm = _llm(cfg)
        rag = _rag(cfg)
        
        response, artifacts = task_conversation.process_user_message(
            task_id=task_id,
            message_content=payload.content,
            llm=llm,
            rag=rag,
            iteration=payload.iteration,
        )
        
        return {
            "status": "ok",
            "assistant_response": response,
            "new_artifacts": artifacts,
            "message_id": f"{task_id}_{payload.iteration}",
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to process message: {str(e)}")


@app.post("/api/tasks/{task_id}/branch")
async def branch_task_api(task_id: str, payload: BranchRequest) -> dict[str, Any]:
    """Create a branched copy of a task."""
    try:
        new_task_id = task_conversation.branch_task(
            task_id=task_id,
            branch_name=payload.branch_name,
            from_iteration=payload.from_iteration,
        )
        return {
            "status": "ok",
            "new_task_id": new_task_id,
            "branch_name": payload.branch_name,
            "parent_id": task_id,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to branch task: {str(e)}")


@app.post("/api/tasks/{task_id}/artifacts/{artifact_id}/re-run")
async def re_run_artifact_api(task_id: str, artifact_id: str, payload: ReRunRequest) -> dict[str, Any]:
    """Re-execute an artifact with edited code."""
    try:
        result = task_conversation.re_execute_artifact(
            task_id=task_id,
            artifact_id=artifact_id,
            edited_code=payload.code,
        )
        return {
            "status": "ok" if result["returncode"] == 0 else "error",
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "returncode": result.get("returncode", -1),
            "execution_time": result.get("execution_time", 0),
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to re-run artifact: {str(e)}")


@app.post("/api/tasks/{task_id}/template")
async def export_template_api(task_id: str) -> dict[str, Any]:
    """Export task as a reusable template."""
    try:
        template_data = task_storage.export_template(task_id)
        # Save template
        templates_dir = Path("output") / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        template_id = str(uuid4())
        template_file = templates_dir / f"{template_id}.json"
        template_file.write_text(json.dumps(template_data, indent=2, default=str), encoding="utf-8")
        
        return {
            "status": "ok",
            "template_id": template_id,
            "task_id": task_id,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to export template: {str(e)}")
