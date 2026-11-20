"""FastAPI application: upload rounds, inspect status, query districts."""
from __future__ import annotations

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import analysis, db, ingest

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_TMP_DIR", str(db.TMP_DIR)))

# Test hooks are compiled in but inert unless explicitly enabled, so a normal
# deployment (the Docker image) exposes no query-holding control surface.
TEST_HOOKS_ENABLED = os.environ.get("ENABLE_TEST_HOOKS", "") == "1"

# Test-only query hold: when enabled, a query blocks just after pinning its
# snapshot pair (via db.snapshot_held_hook) until the holder is released.
# Production deployments never touch these endpoints, so the hook stays inert.
_hold_enabled = threading.Event()
_hold_entered = threading.Event()
_hold_release = threading.Event()


def _maybe_hold_query() -> None:
    if _hold_enabled.is_set():
        _hold_entered.set()
        _hold_release.wait(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(title="公交出行比例两轮重抽分析", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict:
    payload = {"rounds": {}}
    for round_no in (1, 2):
        gen = db.current_generation(round_no)
        if gen is None:
            payload["rounds"][str(round_no)] = {"loaded": False}
            continue
        info = db.generation_info(gen)
        # info: round, created_at, n_samples, n_replicates, n_rep_rows
        payload["rounds"][str(round_no)] = {
            "loaded": True,
            "generation": gen,
            "created_at": info[1].isoformat() if info else None,
            "n_samples": info[2],
            "n_replicates": info[3],
            "n_rep_rows": info[4],
        }
    payload["districts"] = analysis.list_districts()
    return payload


@app.post("/api/rounds/{round_no}")
def upload_round(
    round_no: int,
    sample_csv: UploadFile = File(...),
    replicate_parquet: UploadFile = File(...),
) -> JSONResponse:
    # Sync endpoint: Starlette runs it in the worker thread, and uploads are
    # streamed straight to disk instead of being buffered in memory.
    if round_no not in (1, 2):
        raise HTTPException(status_code=422, detail="轮次必须是 1 或 2")
    if not sample_csv.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="样本文件必须是 .csv")
    if not replicate_parquet.filename.lower().endswith(".parquet"):
        raise HTTPException(status_code=422, detail="重抽权重文件必须是 .parquet")

    upload_dir = tempfile.mkdtemp(prefix=f"round{round_no}_", dir=str(UPLOAD_ROOT))
    try:
        csv_path = ingest.save_upload(upload_dir, sample_csv.filename, sample_csv.file)
        pq_path = ingest.save_upload(
            upload_dir, replicate_parquet.filename, replicate_parquet.file
        )
        try:
            result = ingest.ingest_round(round_no, csv_path, pq_path)
        except ingest.IngestionError as exc:
            # Failed replacement: previously published data remains queryable.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        ingest.remove_upload_dir(upload_dir)

    return JSONResponse({"ok": True, **result})


@app.get("/api/districts")
def districts() -> dict:
    return {"districts": analysis.list_districts()}


@app.get("/api/debug/generations")
def debug_generations() -> dict:
    """Catalog view for lifecycle/ops checks: generations, pins and tables."""
    with db.writer_lock():
        con = db._check()
        generations = [
            {
                "generation": int(g),
                "round": int(r),
                "current": bool(cur),
                "refs": int(refs),
            }
            for g, r, cur, refs in con.execute(
                """
                SELECT g.generation, g.round,
                       EXISTS (SELECT 1 FROM meta_current c
                               WHERE c.generation = g.generation),
                       COALESCE(r.refs, 0)
                FROM meta_generations g
                LEFT JOIN meta_refcount r ON r.generation = g.generation
                ORDER BY g.generation
                """
            ).fetchall()
        ]
        tables = [
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 's\\_gen\\_%' ESCAPE '\\' "
                "   OR table_name LIKE 'r\\_gen\\_%' ESCAPE '\\' "
                "ORDER BY table_name"
            ).fetchall()
        ]
    return {"generations": generations, "tables": tables}


# GET with a query parameter is the natural fit for the browser UI.
@app.get("/api/result")
def result(district: str) -> dict:
    district = (district or "").strip()
    if not district:
        raise HTTPException(status_code=422, detail="请选择一个区")
    try:
        return analysis.compute(district)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ----------------------------------------------------- test-only query hold
def _require_test_hooks() -> None:
    if not TEST_HOOKS_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")


@app.post("/api/test/hold/enable")
def hold_enable() -> dict:
    _require_test_hooks()
    _hold_release.clear()
    _hold_entered.clear()
    _hold_enabled.set()
    db.snapshot_held_hook = _maybe_hold_query
    return {"ok": True}


@app.get("/api/test/hold/entered")
def hold_entered() -> dict:
    _require_test_hooks()
    return {"entered": _hold_entered.is_set()}


@app.post("/api/test/hold/release")
def hold_release() -> dict:
    _require_test_hooks()
    _hold_enabled.clear()
    _hold_release.set()
    db.snapshot_held_hook = None
    return {"ok": True}
